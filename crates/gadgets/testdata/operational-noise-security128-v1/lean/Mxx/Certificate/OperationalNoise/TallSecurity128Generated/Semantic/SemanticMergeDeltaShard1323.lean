import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge215737
def owner : Owner := ⟨.program ⟨257⟩, ⟨15477⟩⟩
def mergeEvent : Nat := 215737
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10209RawTerms
def rightRaw : List Term := Proof.Events810.exact207528RawTerms
def group : MergeGroup := .operator 10209 207528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10209) (leftOrdinal := 0)
    (rightResult := 207528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215737

namespace LeftMerge215742
def owner : Owner := ⟨.program ⟨257⟩, ⟨8610⟩⟩
def mergeEvent : Nat := 215742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }
def leftRaw : List Term := Proof.Events810.exact207398RawTerms
def rightRaw : List Term := Proof.Events099.exact25597RawTerms
def group : MergeGroup := .operator 207398 25597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207398) (leftOrdinal := 0)
    (rightResult := 25597) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215742

namespace LeftMerge215759
def owner : Owner := ⟨.program ⟨257⟩, ⟨15480⟩⟩
def mergeEvent : Nat := 215759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events842.exact215753RawTerms
def rightRaw : List Term := Proof.Events039.exact10212RawTerms
def group : MergeGroup := .operator 215753 10212
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215753) (leftOrdinal := 1)
    (rightResult := 10212) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215759

namespace LeftMerge215760
def owner : Owner := ⟨.program ⟨257⟩, ⟨15480⟩⟩
def mergeEvent : Nat := 215760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }
def leftRaw : List Term := Proof.Events842.exact215753RawTerms
def rightRaw : List Term := Proof.Events039.exact10212RawTerms
def group : MergeGroup := .operator 215753 10212
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215753) (leftOrdinal := 0)
    (rightResult := 10212) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215760

namespace LeftMerge215765
def owner : Owner := ⟨.program ⟨257⟩, ⟨12382⟩⟩
def mergeEvent : Nat := 215765
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10212RawTerms
def rightRaw : List Term := Proof.Events810.exact207528RawTerms
def group : MergeGroup := .operator 10212 207528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10212) (leftOrdinal := 0)
    (rightResult := 207528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215765

namespace LeftMerge215770
def owner : Owner := ⟨.program ⟨257⟩, ⟨8609⟩⟩
def mergeEvent : Nat := 215770
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }
def leftRaw : List Term := Proof.Events810.exact207398RawTerms
def rightRaw : List Term := Proof.Events100.exact25638RawTerms
def group : MergeGroup := .operator 207398 25638
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207398) (leftOrdinal := 0)
    (rightResult := 25638) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215770

namespace LeftMerge215787
def owner : Owner := ⟨.program ⟨257⟩, ⟨12385⟩⟩
def mergeEvent : Nat := 215787
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events842.exact215781RawTerms
def rightRaw : List Term := Proof.Events100.exact25627RawTerms
def group : MergeGroup := .operator 215781 25627
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215781) (leftOrdinal := 1)
    (rightResult := 25627) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215787

namespace LeftMerge215789
def owner : Owner := ⟨.program ⟨257⟩, ⟨12385⟩⟩
def mergeEvent : Nat := 215789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25597RawTerms
def group : MergeGroup := .relation 215788
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215788) (rhsResult := 25597)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9568⟩⟩) ⟨7304⟩ 25597) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215789

namespace LeftMerge215790
def owner : Owner := ⟨.program ⟨257⟩, ⟨12385⟩⟩
def mergeEvent : Nat := 215790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events842.exact215781RawTerms
def rightRaw : List Term := Proof.Events100.exact25627RawTerms
def group : MergeGroup := .operator 215781 25627
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215781) (leftOrdinal := 0)
    (rightResult := 25627) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215790

namespace LeftMerge215795
def owner : Owner := ⟨.program ⟨257⟩, ⟨15481⟩⟩
def mergeEvent : Nat := 215795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }
def leftRaw : List Term := Proof.Events842.exact215791RawTerms
def rightRaw : List Term := Proof.Events842.exact215761RawTerms
def group : MergeGroup := .operator 215791 215761
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215791) (leftOrdinal := 1)
    (rightResult := 215761) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215795

namespace LeftMerge215803
def owner : Owner := ⟨.program ⟨257⟩, ⟨17360⟩⟩
def mergeEvent : Nat := 215803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17359⟩⟩] } }
def leftRaw : List Term := Proof.Events842.exact215797RawTerms
def rightRaw : List Term := Proof.Events842.exact215733RawTerms
def group : MergeGroup := .operator 215797 215733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215797) (leftOrdinal := 1)
    (rightResult := 215733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17359⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17359⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215803

namespace LeftMerge215805
def owner : Owner := ⟨.program ⟨257⟩, ⟨17360⟩⟩
def mergeEvent : Nat := 215805
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16849⟩⟩] } }
def rhsRaw : List Term := Proof.Events842.exact215730RawTerms
def group : MergeGroup := .relation 215804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 215804) (rhsResult := 215730)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17359⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17359⟩⟩) ⟨16849⟩ 215730) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16849⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], [⟨.program ⟨257⟩, ⟨16849⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge215805

namespace LeftMerge215806
def owner : Owner := ⟨.program ⟨257⟩, ⟨17360⟩⟩
def mergeEvent : Nat := 215806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17359⟩⟩] } }
def leftRaw : List Term := Proof.Events842.exact215797RawTerms
def rightRaw : List Term := Proof.Events842.exact215733RawTerms
def group : MergeGroup := .operator 215797 215733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215797) (leftOrdinal := 0)
    (rightResult := 215733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17359⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17359⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215806

namespace LeftMerge215820
def owner : Owner := ⟨.program ⟨257⟩, ⟨16292⟩⟩
def mergeEvent : Nat := 215820
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16289⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events843.exact215814RawTerms
def group : MergeGroup := .operator 207620 215814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 215814) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16289⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16289⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215820

namespace LeftMerge215899
def owner : Owner := ⟨.program ⟨257⟩, ⟨15475⟩⟩
def mergeEvent : Nat := 215899
def frameStart : Nat := 215869
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events843.exact215895RawTerms
def rightRaw : List Term := Proof.Events843.exact215892RawTerms
def group : MergeGroup := .operator 215895 215892
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215895) (leftOrdinal := 0)
    (rightResult := 215892) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12381⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215899

namespace LeftMerge215929
def owner : Owner := ⟨.program ⟨257⟩, ⟨17128⟩⟩
def mergeEvent : Nat := 215929
def frameStart : Nat := 215869
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events843.exact215925RawTerms
def rightRaw : List Term := Proof.Events843.exact215923RawTerms
def group : MergeGroup := .operator 215925 215923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215925) (leftOrdinal := 0)
    (rightResult := 215923) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge215929

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
