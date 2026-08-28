import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge236729
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236729
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37613⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37613⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37613⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236729

namespace LeftMerge236730
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236730
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34933⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34933⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34933⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236730

namespace LeftMerge236731
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236731
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29276⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29276⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236731

namespace LeftMerge236732
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236732
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236732

namespace LeftMerge236733
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236733
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236733

namespace LeftMerge236734
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236734
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63047⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63047⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63047⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236734

namespace LeftMerge236735
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236735
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60067⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60067⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236735

namespace LeftMerge236736
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236736
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57087⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57087⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236736

namespace LeftMerge236737
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236737
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236737

namespace LeftMerge236738
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51127⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51127⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51127⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236738

namespace LeftMerge236739
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236739

namespace LeftMerge236740
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22043⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22043⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22043⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236740

namespace LeftMerge236741
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18823⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18823⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18823⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236741

namespace LeftMerge236742
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def mergeEvent : Nat := 236742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236663RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 236663 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236663) (leftOrdinal := 1)
    (rightResult := 12024) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15998⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15998⟩⟩], [⟨.program ⟨257⟩, ⟨7265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236742

namespace LeftMerge236777
def owner : Owner := ⟨.program ⟨257⟩, ⟨6934⟩⟩
def mergeEvent : Nat := 236777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events924.exact236648RawTerms
def rightRaw : List Term := Proof.Events000.exact2RawTerms
def group : MergeGroup := .operator 236648 2
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236648) (leftOrdinal := 0)
    (rightResult := 2) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236777

namespace LeftMerge236782
def owner : Owner := ⟨.program ⟨257⟩, ⟨47789⟩⟩
def mergeEvent : Nat := 236782
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events044.exact11314RawTerms
def rightRaw : List Term := Proof.Events924.exact236778RawTerms
def group : MergeGroup := .operator 11314 236778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 11314) (leftOrdinal := 0)
    (rightResult := 236778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge236782

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
