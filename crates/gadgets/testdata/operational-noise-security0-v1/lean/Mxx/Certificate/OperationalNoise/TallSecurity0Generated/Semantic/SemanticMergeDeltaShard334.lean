import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge55076
def owner : Owner := ⟨.program ⟨214⟩, ⟨14440⟩⟩
def mergeEvent : Nat := 55076
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55067RawTerms
def rightRaw : List Term := Proof.Events043.exact11011RawTerms
def group : MergeGroup := .operator 55067 11011
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55067) (leftOrdinal := 0)
    (rightResult := 11011) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7855⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55076

namespace LeftMerge55081
def owner : Owner := ⟨.program ⟨214⟩, ⟨14441⟩⟩
def mergeEvent : Nat := 55081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55077RawTerms
def rightRaw : List Term := Proof.Events215.exact55047RawTerms
def group : MergeGroup := .operator 55077 55047
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55077) (leftOrdinal := 1)
    (rightResult := 55047) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55081

namespace LeftMerge55089
def owner : Owner := ⟨.program ⟨214⟩, ⟨26149⟩⟩
def mergeEvent : Nat := 55089
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55083RawTerms
def rightRaw : List Term := Proof.Events214.exact55019RawTerms
def group : MergeGroup := .operator 55083 55019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55083) (leftOrdinal := 1)
    (rightResult := 55019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26148⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55089

namespace LeftMerge55091
def owner : Owner := ⟨.program ⟨214⟩, ⟨26149⟩⟩
def mergeEvent : Nat := 55091
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23628⟩⟩] } }
def rhsRaw : List Term := Proof.Events214.exact55016RawTerms
def group : MergeGroup := .relation 55090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55090) (rhsResult := 55016)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26148⟩⟩) ⟨23628⟩ 55016) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23628⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨23628⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55091

namespace LeftMerge55092
def owner : Owner := ⟨.program ⟨214⟩, ⟨26149⟩⟩
def mergeEvent : Nat := 55092
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55083RawTerms
def rightRaw : List Term := Proof.Events214.exact55019RawTerms
def group : MergeGroup := .operator 55083 55019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55083) (leftOrdinal := 0)
    (rightResult := 55019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26148⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55092

namespace LeftMerge55106
def owner : Owner := ⟨.program ⟨214⟩, ⟨19607⟩⟩
def mergeEvent : Nat := 55106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events215.exact55100RawTerms
def group : MergeGroup := .operator 50762 55100
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 55100) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19604⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55106

namespace LeftMerge55185
def owner : Owner := ⟨.program ⟨214⟩, ⟨14434⟩⟩
def mergeEvent : Nat := 55185
def frameStart : Nat := 55155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events215.exact55181RawTerms
def rightRaw : List Term := Proof.Events215.exact55178RawTerms
def group : MergeGroup := .operator 55181 55178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55181) (leftOrdinal := 0)
    (rightResult := 55178) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11557⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55185

namespace LeftMerge55215
def owner : Owner := ⟨.program ⟨214⟩, ⟨14537⟩⟩
def mergeEvent : Nat := 55215
def frameStart : Nat := 55155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55211RawTerms
def rightRaw : List Term := Proof.Events215.exact55209RawTerms
def group : MergeGroup := .operator 55211 55209
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55211) (leftOrdinal := 0)
    (rightResult := 55209) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55215

namespace LeftMerge55238
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def mergeEvent : Nat := 55238
def frameStart : Nat := 55155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55234RawTerms
def rightRaw : List Term := Proof.Events215.exact55231RawTerms
def group : MergeGroup := .operator 55234 55231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55234) (leftOrdinal := 0)
    (rightResult := 55231) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7855⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55238

namespace LeftMerge55247
def owner : Owner := ⟨.program ⟨214⟩, ⟨26151⟩⟩
def mergeEvent : Nat := 55247
def frameStart : Nat := 55155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55243RawTerms
def rightRaw : List Term := Proof.Events215.exact55200RawTerms
def group : MergeGroup := .operator 55243 55200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55243) (leftOrdinal := 0)
    (rightResult := 55200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26148⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55247

namespace LeftMerge55248
def owner : Owner := ⟨.program ⟨214⟩, ⟨26151⟩⟩
def mergeEvent : Nat := 55248
def frameStart : Nat := 55155
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55243RawTerms
def rightRaw : List Term := Proof.Events215.exact55200RawTerms
def group : MergeGroup := .operator 55243 55200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55243) (leftOrdinal := 1)
    (rightResult := 55200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26148⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55248

namespace LeftMerge55250
def owner : Owner := ⟨.program ⟨214⟩, ⟨26151⟩⟩
def mergeEvent : Nat := 55250
def frameStart : Nat := 55155
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23628⟩⟩] } }
def rhsRaw : List Term := Proof.Events215.exact55197RawTerms
def group : MergeGroup := .relation 55249
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55249) (rhsResult := 55197)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26148⟩⟩) ⟨23628⟩ 55197) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23628⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨23628⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55250

namespace LeftMerge55258
def owner : Owner := ⟨.program ⟨214⟩, ⟨16065⟩⟩
def mergeEvent : Nat := 55258
def frameStart : Nat := 55155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55211RawTerms
def rightRaw : List Term := Proof.Events215.exact55254RawTerms
def group : MergeGroup := .operator 55211 55254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55211) (leftOrdinal := 0)
    (rightResult := 55254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55258

namespace LeftMerge55275
def owner : Owner := ⟨.program ⟨214⟩, ⟨19607⟩⟩
def mergeEvent : Nat := 55275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }
def rhsRaw : List Term := Proof.Events215.exact55272RawTerms
def group : MergeGroup := .relation 55274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55274) (rhsResult := 55272)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55273 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩) (none) 55272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55275

namespace LeftMerge55276
def owner : Owner := ⟨.program ⟨214⟩, ⟨19607⟩⟩
def mergeEvent : Nat := 55276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩] } }
def rhsRaw : List Term := Proof.Events215.exact55272RawTerms
def group : MergeGroup := .relation 55274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55274) (rhsResult := 55272)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55273 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩) (none) 55272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55276

namespace LeftMerge55277
def owner : Owner := ⟨.program ⟨214⟩, ⟨19607⟩⟩
def mergeEvent : Nat := 55277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23628⟩⟩] } }
def rhsRaw : List Term := Proof.Events215.exact55272RawTerms
def group : MergeGroup := .relation 55274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55274) (rhsResult := 55272)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55273 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩) (none) 55272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23628⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨23628⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55277

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
