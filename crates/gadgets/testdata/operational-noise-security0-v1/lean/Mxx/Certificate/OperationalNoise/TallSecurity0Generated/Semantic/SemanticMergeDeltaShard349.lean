import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge57882
def owner : Owner := ⟨.program ⟨214⟩, ⟨20831⟩⟩
def mergeEvent : Nat := 57882
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15426⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23913⟩⟩] } }
def rhsRaw : List Term := Proof.Events226.exact57877RawTerms
def group : MergeGroup := .relation 57879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57879) (rhsResult := 57877)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20828⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 57878 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20828⟩⟩]⟩) (none) 57877) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15426⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23913⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15426⟩⟩], [⟨.program ⟨214⟩, ⟨23913⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57882

namespace LeftMerge57883
def owner : Owner := ⟨.program ⟨214⟩, ⟨20831⟩⟩
def mergeEvent : Nat := 57883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17336⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events226.exact57877RawTerms
def group : MergeGroup := .relation 57879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57879) (rhsResult := 57877)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20828⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 57878 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20828⟩⟩]⟩) (none) 57877) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17336⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17336⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57883

namespace LeftMerge57888
def owner : Owner := ⟨.program ⟨214⟩, ⟨27014⟩⟩
def mergeEvent : Nat := 57888
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27011⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact57884RawTerms
def rightRaw : List Term := Proof.Events225.exact57706RawTerms
def group : MergeGroup := .operator 57884 57706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57884) (leftOrdinal := 0)
    (rightResult := 57706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27011⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27011⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27011⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57888

namespace LeftMerge57889
def owner : Owner := ⟨.program ⟨214⟩, ⟨27014⟩⟩
def mergeEvent : Nat := 57889
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15426⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23913⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact57884RawTerms
def rightRaw : List Term := Proof.Events225.exact57706RawTerms
def group : MergeGroup := .operator 57884 57706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57884) (leftOrdinal := 2)
    (rightResult := 57706) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15426⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23913⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15426⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23913⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15426⟩⟩], [⟨.program ⟨214⟩, ⟨23913⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57889

namespace LeftMerge57915
def owner : Owner := ⟨.program ⟨214⟩, ⟨10988⟩⟩
def mergeEvent : Nat := 57915
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events010.exact2683RawTerms
def rightRaw : List Term := Proof.Events197.exact50670RawTerms
def group : MergeGroup := .operator 2683 50670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2683) (leftOrdinal := 0)
    (rightResult := 50670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57915

namespace LeftMerge57920
def owner : Owner := ⟨.program ⟨214⟩, ⟨7268⟩⟩
def mergeEvent : Nat := 57920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50540RawTerms
def rightRaw : List Term := Proof.Events054.exact13987RawTerms
def group : MergeGroup := .operator 50540 13987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50540) (leftOrdinal := 0)
    (rightResult := 13987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6774⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57920

namespace LeftMerge57937
def owner : Owner := ⟨.program ⟨214⟩, ⟨10991⟩⟩
def mergeEvent : Nat := 57937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact57931RawTerms
def rightRaw : List Term := Proof.Events010.exact2686RawTerms
def group : MergeGroup := .operator 57931 2686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57931) (leftOrdinal := 1)
    (rightResult := 2686) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57937

namespace LeftMerge57938
def owner : Owner := ⟨.program ⟨214⟩, ⟨10991⟩⟩
def mergeEvent : Nat := 57938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact57931RawTerms
def rightRaw : List Term := Proof.Events010.exact2686RawTerms
def group : MergeGroup := .operator 57931 2686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57931) (leftOrdinal := 0)
    (rightResult := 2686) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], [⟨.program ⟨214⟩, ⟨6774⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57938

namespace LeftMerge57943
def owner : Owner := ⟨.program ⟨214⟩, ⟨10848⟩⟩
def mergeEvent : Nat := 57943
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events010.exact2686RawTerms
def rightRaw : List Term := Proof.Events197.exact50670RawTerms
def group : MergeGroup := .operator 2686 50670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2686) (leftOrdinal := 0)
    (rightResult := 50670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57943

namespace LeftMerge57948
def owner : Owner := ⟨.program ⟨214⟩, ⟨7285⟩⟩
def mergeEvent : Nat := 57948
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50540RawTerms
def rightRaw : List Term := Proof.Events054.exact14028RawTerms
def group : MergeGroup := .operator 50540 14028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50540) (leftOrdinal := 0)
    (rightResult := 14028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57948

namespace LeftMerge57965
def owner : Owner := ⟨.program ⟨214⟩, ⟨10851⟩⟩
def mergeEvent : Nat := 57965
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact57959RawTerms
def rightRaw : List Term := Proof.Events054.exact14017RawTerms
def group : MergeGroup := .operator 57959 14017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57959) (leftOrdinal := 1)
    (rightResult := 14017) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7837⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57965

namespace LeftMerge57967
def owner : Owner := ⟨.program ⟨214⟩, ⟨10851⟩⟩
def mergeEvent : Nat := 57967
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }
def rhsRaw : List Term := Proof.Events054.exact13987RawTerms
def group : MergeGroup := .relation 57966
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57966) (rhsResult := 13987)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7837⟩⟩) ⟨6774⟩ 13987) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], [⟨.program ⟨214⟩, ⟨6774⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57967

namespace LeftMerge57968
def owner : Owner := ⟨.program ⟨214⟩, ⟨10851⟩⟩
def mergeEvent : Nat := 57968
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact57959RawTerms
def rightRaw : List Term := Proof.Events054.exact14017RawTerms
def group : MergeGroup := .operator 57959 14017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57959) (leftOrdinal := 0)
    (rightResult := 14017) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7837⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57968

namespace LeftMerge57973
def owner : Owner := ⟨.program ⟨214⟩, ⟨10992⟩⟩
def mergeEvent : Nat := 57973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact57969RawTerms
def rightRaw : List Term := Proof.Events226.exact57939RawTerms
def group : MergeGroup := .operator 57969 57939
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57969) (leftOrdinal := 1)
    (rightResult := 57939) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩], [⟨.program ⟨214⟩, ⟨6774⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57973

namespace LeftMerge57981
def owner : Owner := ⟨.program ⟨214⟩, ⟨25071⟩⟩
def mergeEvent : Nat := 57981
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩] } }
def leftRaw : List Term := Proof.Events226.exact57975RawTerms
def rightRaw : List Term := Proof.Events226.exact57911RawTerms
def group : MergeGroup := .operator 57975 57911
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57975) (leftOrdinal := 1)
    (rightResult := 57911) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25070⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57981

namespace LeftMerge57983
def owner : Owner := ⟨.program ⟨214⟩, ⟨25071⟩⟩
def mergeEvent : Nat := 57983
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23040⟩⟩] } }
def rhsRaw : List Term := Proof.Events226.exact57908RawTerms
def group : MergeGroup := .relation 57982
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57982) (rhsResult := 57908)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25070⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25070⟩⟩) ⟨23040⟩ 57908) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23040⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], [⟨.program ⟨214⟩, ⟨23040⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57983

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
