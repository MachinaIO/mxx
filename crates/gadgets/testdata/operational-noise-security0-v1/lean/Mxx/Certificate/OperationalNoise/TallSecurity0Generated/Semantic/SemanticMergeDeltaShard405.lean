import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge65860
def owner : Owner := ⟨.program ⟨214⟩, ⟨25677⟩⟩
def mergeEvent : Nat := 65860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }
def rhsRaw : List Term := Proof.Events256.exact65785RawTerms
def group : MergeGroup := .relation 65859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65859) (rhsResult := 65785)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25676⟩⟩) ⟨23372⟩ 65785) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨23372⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65860

namespace LeftMerge65861
def owner : Owner := ⟨.program ⟨214⟩, ⟨25677⟩⟩
def mergeEvent : Nat := 65861
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact65852RawTerms
def rightRaw : List Term := Proof.Events256.exact65788RawTerms
def group : MergeGroup := .operator 65852 65788
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65852) (leftOrdinal := 0)
    (rightResult := 65788) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25676⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65861

namespace LeftMerge65875
def owner : Owner := ⟨.program ⟨214⟩, ⟨20175⟩⟩
def mergeEvent : Nat := 65875
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events257.exact65869RawTerms
def group : MergeGroup := .operator 65387 65869
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 65869) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20172⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65875

namespace LeftMerge65954
def owner : Owner := ⟨.program ⟨214⟩, ⟨13147⟩⟩
def mergeEvent : Nat := 65954
def frameStart : Nat := 65924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events257.exact65950RawTerms
def rightRaw : List Term := Proof.Events257.exact65947RawTerms
def group : MergeGroup := .operator 65950 65947
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65950) (leftOrdinal := 0)
    (rightResult := 65947) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65954

namespace LeftMerge65984
def owner : Owner := ⟨.program ⟨214⟩, ⟨13248⟩⟩
def mergeEvent : Nat := 65984
def frameStart : Nat := 65924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact65980RawTerms
def rightRaw : List Term := Proof.Events257.exact65978RawTerms
def group : MergeGroup := .operator 65980 65978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65980) (leftOrdinal := 0)
    (rightResult := 65978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65984

namespace LeftMerge66007
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def mergeEvent : Nat := 66007
def frameStart : Nat := 65924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact66003RawTerms
def rightRaw : List Term := Proof.Events257.exact66000RawTerms
def group : MergeGroup := .operator 66003 66000
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66003) (leftOrdinal := 0)
    (rightResult := 66000) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66007

namespace LeftMerge66016
def owner : Owner := ⟨.program ⟨214⟩, ⟨25679⟩⟩
def mergeEvent : Nat := 66016
def frameStart : Nat := 65924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact66012RawTerms
def rightRaw : List Term := Proof.Events257.exact65969RawTerms
def group : MergeGroup := .operator 66012 65969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66012) (leftOrdinal := 0)
    (rightResult := 65969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25676⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66016

namespace LeftMerge66017
def owner : Owner := ⟨.program ⟨214⟩, ⟨25679⟩⟩
def mergeEvent : Nat := 66017
def frameStart : Nat := 65924
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact66012RawTerms
def rightRaw : List Term := Proof.Events257.exact65969RawTerms
def group : MergeGroup := .operator 66012 65969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66012) (leftOrdinal := 1)
    (rightResult := 65969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25676⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66017

namespace LeftMerge66019
def owner : Owner := ⟨.program ⟨214⟩, ⟨25679⟩⟩
def mergeEvent : Nat := 66019
def frameStart : Nat := 65924
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact65966RawTerms
def group : MergeGroup := .relation 66018
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66018) (rhsResult := 65966)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25676⟩⟩) ⟨23372⟩ 65966) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨23372⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66019

namespace LeftMerge66027
def owner : Owner := ⟨.program ⟨214⟩, ⟨16869⟩⟩
def mergeEvent : Nat := 66027
def frameStart : Nat := 65924
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact65980RawTerms
def rightRaw : List Term := Proof.Events257.exact66023RawTerms
def group : MergeGroup := .operator 65980 66023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65980) (leftOrdinal := 0)
    (rightResult := 66023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66027

namespace LeftMerge66044
def owner : Owner := ⟨.program ⟨214⟩, ⟨20175⟩⟩
def mergeEvent : Nat := 66044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact66041RawTerms
def group : MergeGroup := .relation 66043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66043) (rhsResult := 66041)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66042 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩) (none) 66041) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66044

namespace LeftMerge66045
def owner : Owner := ⟨.program ⟨214⟩, ⟨20175⟩⟩
def mergeEvent : Nat := 66045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact66041RawTerms
def group : MergeGroup := .relation 66043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66043) (rhsResult := 66041)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66042 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩) (none) 66041) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66045

namespace LeftMerge66046
def owner : Owner := ⟨.program ⟨214⟩, ⟨20175⟩⟩
def mergeEvent : Nat := 66046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact66041RawTerms
def group : MergeGroup := .relation 66043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66043) (rhsResult := 66041)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66042 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩) (none) 66041) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨23372⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66046

namespace LeftMerge66047
def owner : Owner := ⟨.program ⟨214⟩, ⟨20175⟩⟩
def mergeEvent : Nat := 66047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact66041RawTerms
def group : MergeGroup := .relation 66043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66043) (rhsResult := 66041)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66042 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩) (none) 66041) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66047

namespace LeftMerge66052
def owner : Owner := ⟨.program ⟨214⟩, ⟨25678⟩⟩
def mergeEvent : Nat := 66052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }
def leftRaw : List Term := Proof.Events258.exact66048RawTerms
def rightRaw : List Term := Proof.Events257.exact65862RawTerms
def group : MergeGroup := .operator 66048 65862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66048) (leftOrdinal := 2)
    (rightResult := 65862) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨23372⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66052

namespace LeftMerge66053
def owner : Owner := ⟨.program ⟨214⟩, ⟨25678⟩⟩
def mergeEvent : Nat := 66053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }
def leftRaw : List Term := Proof.Events258.exact66048RawTerms
def rightRaw : List Term := Proof.Events257.exact65862RawTerms
def group : MergeGroup := .operator 66048 65862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66048) (leftOrdinal := 1)
    (rightResult := 65862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66053

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
