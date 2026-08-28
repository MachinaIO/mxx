import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge222101
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222101
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222101

namespace LeftMerge222102
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222102
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42989⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42989⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222102

namespace LeftMerge222103
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222103
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222103

namespace LeftMerge222104
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222104
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37626⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37626⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222104

namespace LeftMerge222105
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34946⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34946⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222105

namespace LeftMerge222106
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29289⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29289⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29289⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222106

namespace LeftMerge222107
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26609⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26609⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26609⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222107

namespace LeftMerge222108
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222108
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66518⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66518⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222108

namespace LeftMerge222109
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222109
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63066⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63066⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222109

namespace LeftMerge222110
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222110
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60086⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60086⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222110

namespace LeftMerge222111
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222111
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57106⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57106⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222111

namespace LeftMerge222112
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222112
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54126⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54126⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222112

namespace LeftMerge222113
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222113
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51146⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51146⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51146⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222113

namespace LeftMerge222114
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222114
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32082⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32082⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222114

namespace LeftMerge222115
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222115

namespace LeftMerge222116
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def mergeEvent : Nat := 222116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222038RawTerms
def rightRaw : List Term := Proof.Events044.exact11276RawTerms
def group : MergeGroup := .operator 222038 11276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222038) (leftOrdinal := 1)
    (rightResult := 11276) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18842⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18842⟩⟩], [⟨.program ⟨257⟩, ⟨7263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222116

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
