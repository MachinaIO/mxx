import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge40948
def owner : Owner := ⟨.program ⟨214⟩, ⟨26077⟩⟩
def mergeEvent : Nat := 40948
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23588⟩⟩] } }
def rhsRaw : List Term := Proof.Events159.exact40873RawTerms
def group : MergeGroup := .relation 40947
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40947) (rhsResult := 40873)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26076⟩⟩) ⟨23588⟩ 40873) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23588⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], [⟨.program ⟨214⟩, ⟨23588⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40948

namespace LeftMerge40949
def owner : Owner := ⟨.program ⟨214⟩, ⟨26077⟩⟩
def mergeEvent : Nat := 40949
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40940RawTerms
def rightRaw : List Term := Proof.Events159.exact40876RawTerms
def group : MergeGroup := .operator 40940 40876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40940) (leftOrdinal := 0)
    (rightResult := 40876) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26076⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40949

namespace LeftMerge40963
def owner : Owner := ⟨.program ⟨214⟩, ⟨19539⟩⟩
def mergeEvent : Nat := 40963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events159.exact40957RawTerms
def group : MergeGroup := .operator 36137 40957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 40957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19536⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40963

namespace LeftMerge41042
def owner : Owner := ⟨.program ⟨214⟩, ⟨14226⟩⟩
def mergeEvent : Nat := 41042
def frameStart : Nat := 41012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events160.exact41038RawTerms
def rightRaw : List Term := Proof.Events160.exact41035RawTerms
def group : MergeGroup := .operator 41038 41035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41038) (leftOrdinal := 0)
    (rightResult := 41035) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11477⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41042

namespace LeftMerge41072
def owner : Owner := ⟨.program ⟨214⟩, ⟨14324⟩⟩
def mergeEvent : Nat := 41072
def frameStart : Nat := 41012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events160.exact41068RawTerms
def rightRaw : List Term := Proof.Events160.exact41066RawTerms
def group : MergeGroup := .operator 41068 41066
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41068) (leftOrdinal := 0)
    (rightResult := 41066) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41072

namespace LeftMerge41095
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def mergeEvent : Nat := 41095
def frameStart : Nat := 41012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }
def leftRaw : List Term := Proof.Events160.exact41091RawTerms
def rightRaw : List Term := Proof.Events160.exact41088RawTerms
def group : MergeGroup := .operator 41091 41088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41091) (leftOrdinal := 0)
    (rightResult := 41088) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7852⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41095

namespace LeftMerge41104
def owner : Owner := ⟨.program ⟨214⟩, ⟨26079⟩⟩
def mergeEvent : Nat := 41104
def frameStart : Nat := 41012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩] } }
def leftRaw : List Term := Proof.Events160.exact41100RawTerms
def rightRaw : List Term := Proof.Events160.exact41057RawTerms
def group : MergeGroup := .operator 41100 41057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41100) (leftOrdinal := 0)
    (rightResult := 41057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26076⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41104

namespace LeftMerge41105
def owner : Owner := ⟨.program ⟨214⟩, ⟨26079⟩⟩
def mergeEvent : Nat := 41105
def frameStart : Nat := 41012
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩] } }
def leftRaw : List Term := Proof.Events160.exact41100RawTerms
def rightRaw : List Term := Proof.Events160.exact41057RawTerms
def group : MergeGroup := .operator 41100 41057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41100) (leftOrdinal := 1)
    (rightResult := 41057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26076⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41105

namespace LeftMerge41107
def owner : Owner := ⟨.program ⟨214⟩, ⟨26079⟩⟩
def mergeEvent : Nat := 41107
def frameStart : Nat := 41012
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23588⟩⟩] } }
def rhsRaw : List Term := Proof.Events160.exact41054RawTerms
def group : MergeGroup := .relation 41106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41106) (rhsResult := 41054)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26076⟩⟩) ⟨23588⟩ 41054) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23588⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], [⟨.program ⟨214⟩, ⟨23588⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41107

namespace LeftMerge41115
def owner : Owner := ⟨.program ⟨214⟩, ⟨15950⟩⟩
def mergeEvent : Nat := 41115
def frameStart : Nat := 41012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15948⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events160.exact41068RawTerms
def rightRaw : List Term := Proof.Events160.exact41111RawTerms
def group : MergeGroup := .operator 41068 41111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41068) (leftOrdinal := 0)
    (rightResult := 41111) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15948⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41115

namespace LeftMerge41132
def owner : Owner := ⟨.program ⟨214⟩, ⟨19539⟩⟩
def mergeEvent : Nat := 41132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }
def rhsRaw : List Term := Proof.Events160.exact41129RawTerms
def group : MergeGroup := .relation 41131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41131) (rhsResult := 41129)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 41130 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩) (none) 41129) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41132

namespace LeftMerge41133
def owner : Owner := ⟨.program ⟨214⟩, ⟨19539⟩⟩
def mergeEvent : Nat := 41133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩] } }
def rhsRaw : List Term := Proof.Events160.exact41129RawTerms
def group : MergeGroup := .relation 41131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41131) (rhsResult := 41129)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 41130 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩) (none) 41129) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41133

namespace LeftMerge41134
def owner : Owner := ⟨.program ⟨214⟩, ⟨19539⟩⟩
def mergeEvent : Nat := 41134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23588⟩⟩] } }
def rhsRaw : List Term := Proof.Events160.exact41129RawTerms
def group : MergeGroup := .relation 41131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41131) (rhsResult := 41129)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 41130 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩) (none) 41129) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23588⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], [⟨.program ⟨214⟩, ⟨23588⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41134

namespace LeftMerge41135
def owner : Owner := ⟨.program ⟨214⟩, ⟨19539⟩⟩
def mergeEvent : Nat := 41135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events160.exact41129RawTerms
def group : MergeGroup := .relation 41131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 41131) (rhsResult := 41129)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 41130 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩) (none) 41129) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15948⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41135

namespace LeftMerge41140
def owner : Owner := ⟨.program ⟨214⟩, ⟨26078⟩⟩
def mergeEvent : Nat := 41140
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23588⟩⟩] } }
def leftRaw : List Term := Proof.Events160.exact41136RawTerms
def rightRaw : List Term := Proof.Events159.exact40950RawTerms
def group : MergeGroup := .operator 41136 40950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41136) (leftOrdinal := 2)
    (rightResult := 40950) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23588⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23588⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], [⟨.program ⟨214⟩, ⟨23588⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge41140

namespace LeftMerge41141
def owner : Owner := ⟨.program ⟨214⟩, ⟨26078⟩⟩
def mergeEvent : Nat := 41141
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩] } }
def leftRaw : List Term := Proof.Events160.exact41136RawTerms
def rightRaw : List Term := Proof.Events159.exact40950RawTerms
def group : MergeGroup := .operator 41136 40950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41136) (leftOrdinal := 1)
    (rightResult := 40950) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26076⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge41141

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
