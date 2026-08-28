import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge24859
def owner : Owner := ⟨.program ⟨214⟩, ⟨9629⟩⟩
def mergeEvent : Nat := 24859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24853RawTerms
def rightRaw : List Term := Proof.Events039.exact10009RawTerms
def group : MergeGroup := .operator 24853 10009
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24853) (leftOrdinal := 1)
    (rightResult := 10009) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7861⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24859

namespace LeftMerge24861
def owner : Owner := ⟨.program ⟨214⟩, ⟨9629⟩⟩
def mergeEvent : Nat := 24861
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }
def rhsRaw : List Term := Proof.Events038.exact9979RawTerms
def group : MergeGroup := .relation 24860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24860) (rhsResult := 9979)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7861⟩⟩) ⟨6783⟩ 9979) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24861

namespace LeftMerge24862
def owner : Owner := ⟨.program ⟨214⟩, ⟨9629⟩⟩
def mergeEvent : Nat := 24862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24853RawTerms
def rightRaw : List Term := Proof.Events039.exact10009RawTerms
def group : MergeGroup := .operator 24853 10009
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24853) (leftOrdinal := 0)
    (rightResult := 10009) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7861⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24862

namespace LeftMerge24867
def owner : Owner := ⟨.program ⟨214⟩, ⟨11792⟩⟩
def mergeEvent : Nat := 24867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24863RawTerms
def rightRaw : List Term := Proof.Events097.exact24833RawTerms
def group : MergeGroup := .operator 24863 24833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24863) (leftOrdinal := 1)
    (rightResult := 24833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24867

namespace LeftMerge24875
def owner : Owner := ⟨.program ⟨214⟩, ⟨25158⟩⟩
def mergeEvent : Nat := 24875
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24869RawTerms
def rightRaw : List Term := Proof.Events096.exact24805RawTerms
def group : MergeGroup := .operator 24869 24805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24869) (leftOrdinal := 1)
    (rightResult := 24805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25157⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24875

namespace LeftMerge24877
def owner : Owner := ⟨.program ⟨214⟩, ⟨25158⟩⟩
def mergeEvent : Nat := 24877
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24802RawTerms
def group : MergeGroup := .relation 24876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24876) (rhsResult := 24802)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25157⟩⟩) ⟨23086⟩ 24802) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨23086⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24877

namespace LeftMerge24878
def owner : Owner := ⟨.program ⟨214⟩, ⟨25158⟩⟩
def mergeEvent : Nat := 24878
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24869RawTerms
def rightRaw : List Term := Proof.Events096.exact24805RawTerms
def group : MergeGroup := .operator 24869 24805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24869) (leftOrdinal := 0)
    (rightResult := 24805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25157⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24878

namespace LeftMerge24892
def owner : Owner := ⟨.program ⟨214⟩, ⟨19759⟩⟩
def mergeEvent : Nat := 24892
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events097.exact24886RawTerms
def group : MergeGroup := .operator 21512 24886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 24886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24892

namespace LeftMerge24971
def owner : Owner := ⟨.program ⟨214⟩, ⟨11786⟩⟩
def mergeEvent : Nat := 24971
def frameStart : Nat := 24941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events097.exact24967RawTerms
def rightRaw : List Term := Proof.Events097.exact24964RawTerms
def group : MergeGroup := .operator 24967 24964
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24967) (leftOrdinal := 0)
    (rightResult := 24964) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24971

namespace LeftMerge25001
def owner : Owner := ⟨.program ⟨214⟩, ⟨11871⟩⟩
def mergeEvent : Nat := 25001
def frameStart : Nat := 24941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24997RawTerms
def rightRaw : List Term := Proof.Events097.exact24995RawTerms
def group : MergeGroup := .operator 24997 24995
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24997) (leftOrdinal := 0)
    (rightResult := 24995) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25001

namespace LeftMerge25024
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def mergeEvent : Nat := 25024
def frameStart : Nat := 24941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact25020RawTerms
def rightRaw : List Term := Proof.Events097.exact25017RawTerms
def group : MergeGroup := .operator 25020 25017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25020) (leftOrdinal := 0)
    (rightResult := 25017) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7861⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25024

namespace LeftMerge25033
def owner : Owner := ⟨.program ⟨214⟩, ⟨25160⟩⟩
def mergeEvent : Nat := 25033
def frameStart : Nat := 24941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact25029RawTerms
def rightRaw : List Term := Proof.Events097.exact24986RawTerms
def group : MergeGroup := .operator 25029 24986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25029) (leftOrdinal := 0)
    (rightResult := 24986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25157⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25033

namespace LeftMerge25034
def owner : Owner := ⟨.program ⟨214⟩, ⟨25160⟩⟩
def mergeEvent : Nat := 25034
def frameStart : Nat := 24941
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact25029RawTerms
def rightRaw : List Term := Proof.Events097.exact24986RawTerms
def group : MergeGroup := .operator 25029 24986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25029) (leftOrdinal := 1)
    (rightResult := 24986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25157⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25034

namespace LeftMerge25036
def owner : Owner := ⟨.program ⟨214⟩, ⟨25160⟩⟩
def mergeEvent : Nat := 25036
def frameStart : Nat := 24941
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact24983RawTerms
def group : MergeGroup := .relation 25035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25035) (rhsResult := 24983)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25157⟩⟩) ⟨23086⟩ 24983) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨23086⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25036

namespace LeftMerge25044
def owner : Owner := ⟨.program ⟨214⟩, ⟨16276⟩⟩
def mergeEvent : Nat := 25044
def frameStart : Nat := 24941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24997RawTerms
def rightRaw : List Term := Proof.Events097.exact25040RawTerms
def group : MergeGroup := .operator 24997 25040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24997) (leftOrdinal := 0)
    (rightResult := 25040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25044

namespace LeftMerge25061
def owner : Owner := ⟨.program ⟨214⟩, ⟨19759⟩⟩
def mergeEvent : Nat := 25061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact25058RawTerms
def group : MergeGroup := .relation 25060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25060) (rhsResult := 25058)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25059 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩]⟩) (none) 25058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25061

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
