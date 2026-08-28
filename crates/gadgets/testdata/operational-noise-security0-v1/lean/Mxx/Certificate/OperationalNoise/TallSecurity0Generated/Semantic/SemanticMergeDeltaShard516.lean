import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge84028
def owner : Owner := ⟨.program ⟨214⟩, ⟨19675⟩⟩
def mergeEvent : Nat := 84028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84022RawTerms
def group : MergeGroup := .relation 84024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84024) (rhsResult := 84022)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84023 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩) (none) 84022) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84028

namespace LeftMerge84033
def owner : Owner := ⟨.program ⟨214⟩, ⟨26222⟩⟩
def mergeEvent : Nat := 84033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23668⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84029RawTerms
def rightRaw : List Term := Proof.Events327.exact83845RawTerms
def group : MergeGroup := .operator 84029 83845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84029) (leftOrdinal := 2)
    (rightResult := 83845) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23668⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23668⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84033

namespace LeftMerge84034
def owner : Owner := ⟨.program ⟨214⟩, ⟨26222⟩⟩
def mergeEvent : Nat := 84034
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84029RawTerms
def rightRaw : List Term := Proof.Events327.exact83845RawTerms
def group : MergeGroup := .operator 84029 83845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84029) (leftOrdinal := 1)
    (rightResult := 83845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84034

namespace LeftMerge84042
def owner : Owner := ⟨.program ⟨214⟩, ⟨28302⟩⟩
def mergeEvent : Nat := 84042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84036RawTerms
def rightRaw : List Term := Proof.Events327.exact83761RawTerms
def group : MergeGroup := .operator 84036 83761
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84036) (leftOrdinal := 0)
    (rightResult := 83761) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28300⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84042

namespace LeftMerge84043
def owner : Owner := ⟨.program ⟨214⟩, ⟨28302⟩⟩
def mergeEvent : Nat := 84043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84036RawTerms
def rightRaw : List Term := Proof.Events327.exact83761RawTerms
def group : MergeGroup := .operator 84036 83761
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84036) (leftOrdinal := 1)
    (rightResult := 83761) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28300⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84043

namespace LeftMerge84045
def owner : Owner := ⟨.program ⟨214⟩, ⟨28302⟩⟩
def mergeEvent : Nat := 84045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24288⟩⟩] } }
def rhsRaw : List Term := Proof.Events327.exact83758RawTerms
def group : MergeGroup := .relation 84044
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84044) (rhsResult := 83758)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28300⟩⟩) ⟨24288⟩ 83758) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24288⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84045

namespace LeftMerge84059
def owner : Owner := ⟨.program ⟨214⟩, ⟨21691⟩⟩
def mergeEvent : Nat := 84059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events328.exact84053RawTerms
def group : MergeGroup := .operator 80012 84053
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 84053) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21688⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84059

namespace LeftMerge84180
def owner : Owner := ⟨.program ⟨214⟩, ⟨16220⟩⟩
def mergeEvent : Nat := 84180
def frameStart : Nat := 84114
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84176RawTerms
def rightRaw : List Term := Proof.Events328.exact84174RawTerms
def group : MergeGroup := .operator 84176 84174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84176) (leftOrdinal := 0)
    (rightResult := 84174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84180

namespace LeftMerge84192
def owner : Owner := ⟨.program ⟨214⟩, ⟨28301⟩⟩
def mergeEvent : Nat := 84192
def frameStart : Nat := 84114
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84188RawTerms
def rightRaw : List Term := Proof.Events328.exact84165RawTerms
def group : MergeGroup := .operator 84188 84165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84188) (leftOrdinal := 0)
    (rightResult := 84165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28300⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84192

namespace LeftMerge84193
def owner : Owner := ⟨.program ⟨214⟩, ⟨28301⟩⟩
def mergeEvent : Nat := 84193
def frameStart : Nat := 84114
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84188RawTerms
def rightRaw : List Term := Proof.Events328.exact84165RawTerms
def group : MergeGroup := .operator 84188 84165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84188) (leftOrdinal := 1)
    (rightResult := 84165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28300⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84193

namespace LeftMerge84195
def owner : Owner := ⟨.program ⟨214⟩, ⟨28301⟩⟩
def mergeEvent : Nat := 84195
def frameStart : Nat := 84114
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24288⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84162RawTerms
def group : MergeGroup := .relation 84194
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84194) (rhsResult := 84162)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28300⟩⟩) ⟨24288⟩ 84162) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24288⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84195

namespace LeftMerge84203
def owner : Owner := ⟨.program ⟨214⟩, ⟨18351⟩⟩
def mergeEvent : Nat := 84203
def frameStart : Nat := 84114
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events328.exact84176RawTerms
def rightRaw : List Term := Proof.Events328.exact84199RawTerms
def group : MergeGroup := .operator 84176 84199
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84176) (leftOrdinal := 0)
    (rightResult := 84199) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84203

namespace LeftMerge84220
def owner : Owner := ⟨.program ⟨214⟩, ⟨21691⟩⟩
def mergeEvent : Nat := 84220
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84217RawTerms
def group : MergeGroup := .relation 84219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84219) (rhsResult := 84217)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84218 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩) (none) 84217) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84220

namespace LeftMerge84221
def owner : Owner := ⟨.program ⟨214⟩, ⟨21691⟩⟩
def mergeEvent : Nat := 84221
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84217RawTerms
def group : MergeGroup := .relation 84219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84219) (rhsResult := 84217)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84218 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩) (none) 84217) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84221

namespace LeftMerge84222
def owner : Owner := ⟨.program ⟨214⟩, ⟨21691⟩⟩
def mergeEvent : Nat := 84222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24288⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84217RawTerms
def group : MergeGroup := .relation 84219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84219) (rhsResult := 84217)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84218 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩) (none) 84217) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24288⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84222

namespace LeftMerge84223
def owner : Owner := ⟨.program ⟨214⟩, ⟨21691⟩⟩
def mergeEvent : Nat := 84223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events328.exact84217RawTerms
def group : MergeGroup := .relation 84219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84219) (rhsResult := 84217)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 84218 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩) (none) 84217) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84223

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
