import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge128605
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128605
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 18)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128605

namespace LeftMerge128607
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events467.exact119750RawTerms
def group : MergeGroup := .relation 128606
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128606) (rhsResult := 119750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 119750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128607

namespace LeftMerge128621
def owner : Owner := ⟨.program ⟨257⟩, ⟨68333⟩⟩
def mergeEvent : Nat := 128621
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events502.exact128615RawTerms
def group : MergeGroup := .operator 119870 128615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 128615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68330⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128621

namespace LeftMerge129742
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129742
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48311⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129742

namespace LeftMerge129743
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129743
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45631⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45631⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45631⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129743

namespace LeftMerge129744
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129744
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42947⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42947⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42947⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129744

namespace LeftMerge129745
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129745
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40267⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40267⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129745

namespace LeftMerge129746
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129746
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37591⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37591⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37591⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129746

namespace LeftMerge129747
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129747
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34911⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129747

namespace LeftMerge129748
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129748
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29247⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29247⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29247⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129748

namespace LeftMerge129749
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129749
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26567⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26567⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26567⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129749

namespace LeftMerge129750
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129750
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66321⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66321⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66321⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129750

namespace LeftMerge129751
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129751
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63005⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63005⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129751

namespace LeftMerge129752
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129752
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60025⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129752

namespace LeftMerge129753
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129753
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57045⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57045⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129753

namespace LeftMerge129754
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def mergeEvent : Nat := 129754
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54065⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events506.exact129736RawTerms
def group : MergeGroup := .operator 129738 129736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129736) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54065⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129754

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
