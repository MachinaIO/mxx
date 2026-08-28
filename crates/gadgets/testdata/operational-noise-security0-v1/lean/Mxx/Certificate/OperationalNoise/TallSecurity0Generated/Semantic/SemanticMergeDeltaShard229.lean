import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge38040
def owner : Owner := ⟨.program ⟨214⟩, ⟨9939⟩⟩
def mergeEvent : Nat := 38040
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }
def rhsRaw : List Term := Proof.Events033.exact8476RawTerms
def group : MergeGroup := .relation 38039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38039) (rhsResult := 8476)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7870⟩⟩) ⟨6786⟩ 8476) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩], [⟨.program ⟨214⟩, ⟨6786⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38040

namespace LeftMerge38041
def owner : Owner := ⟨.program ⟨214⟩, ⟨9939⟩⟩
def mergeEvent : Nat := 38041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38032RawTerms
def rightRaw : List Term := Proof.Events033.exact8506RawTerms
def group : MergeGroup := .operator 38032 8506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38032) (leftOrdinal := 0)
    (rightResult := 8506) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7870⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38041

namespace LeftMerge38046
def owner : Owner := ⟨.program ⟨214⟩, ⟨12589⟩⟩
def mergeEvent : Nat := 38046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38042RawTerms
def rightRaw : List Term := Proof.Events148.exact38012RawTerms
def group : MergeGroup := .operator 38042 38012
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38042) (leftOrdinal := 1)
    (rightResult := 38012) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩], [⟨.program ⟨214⟩, ⟨6786⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38046

namespace LeftMerge38054
def owner : Owner := ⟨.program ⟨214⟩, ⟨25461⟩⟩
def mergeEvent : Nat := 38054
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38048RawTerms
def rightRaw : List Term := Proof.Events148.exact37984RawTerms
def group : MergeGroup := .operator 38048 37984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38048) (leftOrdinal := 1)
    (rightResult := 37984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25460⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38054

namespace LeftMerge38056
def owner : Owner := ⟨.program ⟨214⟩, ⟨25461⟩⟩
def mergeEvent : Nat := 38056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23252⟩⟩] } }
def rhsRaw : List Term := Proof.Events148.exact37981RawTerms
def group : MergeGroup := .relation 38055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38055) (rhsResult := 37981)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25460⟩⟩) ⟨23252⟩ 37981) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23252⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], [⟨.program ⟨214⟩, ⟨23252⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38056

namespace LeftMerge38057
def owner : Owner := ⟨.program ⟨214⟩, ⟨25461⟩⟩
def mergeEvent : Nat := 38057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38048RawTerms
def rightRaw : List Term := Proof.Events148.exact37984RawTerms
def group : MergeGroup := .operator 38048 37984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38048) (leftOrdinal := 0)
    (rightResult := 37984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25460⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38057

namespace LeftMerge38071
def owner : Owner := ⟨.program ⟨214⟩, ⟨19971⟩⟩
def mergeEvent : Nat := 38071
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events148.exact38065RawTerms
def group : MergeGroup := .operator 36137 38065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 38065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19968⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38071

namespace LeftMerge38150
def owner : Owner := ⟨.program ⟨214⟩, ⟨12583⟩⟩
def mergeEvent : Nat := 38150
def frameStart : Nat := 38120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events149.exact38146RawTerms
def rightRaw : List Term := Proof.Events148.exact38143RawTerms
def group : MergeGroup := .operator 38146 38143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38146) (leftOrdinal := 0)
    (rightResult := 38143) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9935⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38150

namespace LeftMerge38180
def owner : Owner := ⟨.program ⟨214⟩, ⟨12672⟩⟩
def mergeEvent : Nat := 38180
def frameStart : Nat := 38120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38176RawTerms
def rightRaw : List Term := Proof.Events149.exact38174RawTerms
def group : MergeGroup := .operator 38176 38174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38176) (leftOrdinal := 0)
    (rightResult := 38174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38180

namespace LeftMerge38203
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def mergeEvent : Nat := 38203
def frameStart : Nat := 38120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38199RawTerms
def rightRaw : List Term := Proof.Events149.exact38196RawTerms
def group : MergeGroup := .operator 38199 38196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38199) (leftOrdinal := 0)
    (rightResult := 38196) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7870⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38203

namespace LeftMerge38212
def owner : Owner := ⟨.program ⟨214⟩, ⟨25463⟩⟩
def mergeEvent : Nat := 38212
def frameStart : Nat := 38120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38208RawTerms
def rightRaw : List Term := Proof.Events149.exact38165RawTerms
def group : MergeGroup := .operator 38208 38165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38208) (leftOrdinal := 0)
    (rightResult := 38165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25460⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38212

namespace LeftMerge38213
def owner : Owner := ⟨.program ⟨214⟩, ⟨25463⟩⟩
def mergeEvent : Nat := 38213
def frameStart : Nat := 38120
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38208RawTerms
def rightRaw : List Term := Proof.Events149.exact38165RawTerms
def group : MergeGroup := .operator 38208 38165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38208) (leftOrdinal := 1)
    (rightResult := 38165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25460⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38213

namespace LeftMerge38215
def owner : Owner := ⟨.program ⟨214⟩, ⟨25463⟩⟩
def mergeEvent : Nat := 38215
def frameStart : Nat := 38120
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23252⟩⟩] } }
def rhsRaw : List Term := Proof.Events149.exact38162RawTerms
def group : MergeGroup := .relation 38214
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38214) (rhsResult := 38162)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25460⟩⟩) ⟨23252⟩ 38162) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23252⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], [⟨.program ⟨214⟩, ⟨23252⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38215

namespace LeftMerge38223
def owner : Owner := ⟨.program ⟨214⟩, ⟨16559⟩⟩
def mergeEvent : Nat := 38223
def frameStart : Nat := 38120
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38176RawTerms
def rightRaw : List Term := Proof.Events149.exact38219RawTerms
def group : MergeGroup := .operator 38176 38219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38176) (leftOrdinal := 0)
    (rightResult := 38219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38223

namespace LeftMerge38240
def owner : Owner := ⟨.program ⟨214⟩, ⟨19971⟩⟩
def mergeEvent : Nat := 38240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }
def rhsRaw : List Term := Proof.Events149.exact38237RawTerms
def group : MergeGroup := .relation 38239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38239) (rhsResult := 38237)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38238 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩) (none) 38237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38240

namespace LeftMerge38241
def owner : Owner := ⟨.program ⟨214⟩, ⟨19971⟩⟩
def mergeEvent : Nat := 38241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩] } }
def rhsRaw : List Term := Proof.Events149.exact38237RawTerms
def group : MergeGroup := .relation 38239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38239) (rhsResult := 38237)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38238 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩) (none) 38237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38241

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
