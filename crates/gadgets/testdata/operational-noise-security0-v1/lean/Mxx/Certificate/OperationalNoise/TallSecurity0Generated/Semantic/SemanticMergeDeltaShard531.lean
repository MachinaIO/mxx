import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge86724
def owner : Owner := ⟨.program ⟨214⟩, ⟨25297⟩⟩
def mergeEvent : Nat := 86724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩] } }
def leftRaw : List Term := Proof.Events338.exact86715RawTerms
def rightRaw : List Term := Proof.Events338.exact86651RawTerms
def group : MergeGroup := .operator 86715 86651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86715) (leftOrdinal := 0)
    (rightResult := 86651) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25296⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86724

namespace LeftMerge86738
def owner : Owner := ⟨.program ⟨214⟩, ⟨19243⟩⟩
def mergeEvent : Nat := 86738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events338.exact86732RawTerms
def group : MergeGroup := .operator 80012 86732
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 86732) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19240⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86738

namespace LeftMerge86817
def owner : Owner := ⟨.program ⟨214⟩, ⟨12164⟩⟩
def mergeEvent : Nat := 86817
def frameStart : Nat := 86787
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events339.exact86813RawTerms
def rightRaw : List Term := Proof.Events339.exact86810RawTerms
def group : MergeGroup := .operator 86813 86810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86813) (leftOrdinal := 0)
    (rightResult := 86810) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86817

namespace LeftMerge86847
def owner : Owner := ⟨.program ⟨214⟩, ⟨12272⟩⟩
def mergeEvent : Nat := 86847
def frameStart : Nat := 86787
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86843RawTerms
def rightRaw : List Term := Proof.Events339.exact86841RawTerms
def group : MergeGroup := .operator 86843 86841
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86843) (leftOrdinal := 0)
    (rightResult := 86841) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86847

namespace LeftMerge86868
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def mergeEvent : Nat := 86868
def frameStart : Nat := 86787
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86864RawTerms
def rightRaw : List Term := Proof.Events339.exact86861RawTerms
def group : MergeGroup := .operator 86864 86861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86864) (leftOrdinal := 0)
    (rightResult := 86861) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86868

namespace LeftMerge86877
def owner : Owner := ⟨.program ⟨214⟩, ⟨25299⟩⟩
def mergeEvent : Nat := 86877
def frameStart : Nat := 86787
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86873RawTerms
def rightRaw : List Term := Proof.Events339.exact86832RawTerms
def group : MergeGroup := .operator 86873 86832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86873) (leftOrdinal := 0)
    (rightResult := 86832) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25296⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86877

namespace LeftMerge86878
def owner : Owner := ⟨.program ⟨214⟩, ⟨25299⟩⟩
def mergeEvent : Nat := 86878
def frameStart : Nat := 86787
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86873RawTerms
def rightRaw : List Term := Proof.Events339.exact86832RawTerms
def group : MergeGroup := .operator 86873 86832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86873) (leftOrdinal := 1)
    (rightResult := 86832) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25296⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86878

namespace LeftMerge86880
def owner : Owner := ⟨.program ⟨214⟩, ⟨25299⟩⟩
def mergeEvent : Nat := 86880
def frameStart : Nat := 86787
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23164⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86829RawTerms
def group : MergeGroup := .relation 86879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86879) (rhsResult := 86829)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25296⟩⟩) ⟨23164⟩ 86829) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23164⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86880

namespace LeftMerge86888
def owner : Owner := ⟨.program ⟨214⟩, ⟨15424⟩⟩
def mergeEvent : Nat := 86888
def frameStart : Nat := 86787
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15422⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86843RawTerms
def rightRaw : List Term := Proof.Events339.exact86884RawTerms
def group : MergeGroup := .operator 86843 86884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86843) (leftOrdinal := 0)
    (rightResult := 86884) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15422⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86888

namespace LeftMerge86905
def owner : Owner := ⟨.program ⟨214⟩, ⟨19243⟩⟩
def mergeEvent : Nat := 86905
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86902RawTerms
def group : MergeGroup := .relation 86904
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86904) (rhsResult := 86902)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86903 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩) (none) 86902) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86905

namespace LeftMerge86906
def owner : Owner := ⟨.program ⟨214⟩, ⟨19243⟩⟩
def mergeEvent : Nat := 86906
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86902RawTerms
def group : MergeGroup := .relation 86904
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86904) (rhsResult := 86902)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86903 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩) (none) 86902) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86906

namespace LeftMerge86907
def owner : Owner := ⟨.program ⟨214⟩, ⟨19243⟩⟩
def mergeEvent : Nat := 86907
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23164⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86902RawTerms
def group : MergeGroup := .relation 86904
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86904) (rhsResult := 86902)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86903 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩) (none) 86902) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23164⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86907

namespace LeftMerge86908
def owner : Owner := ⟨.program ⟨214⟩, ⟨19243⟩⟩
def mergeEvent : Nat := 86908
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86902RawTerms
def group : MergeGroup := .relation 86904
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86904) (rhsResult := 86902)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86903 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩) (none) 86902) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15422⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86908

namespace LeftMerge86913
def owner : Owner := ⟨.program ⟨214⟩, ⟨25298⟩⟩
def mergeEvent : Nat := 86913
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23164⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86909RawTerms
def rightRaw : List Term := Proof.Events338.exact86725RawTerms
def group : MergeGroup := .operator 86909 86725
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86909) (leftOrdinal := 2)
    (rightResult := 86725) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23164⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23164⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86913

namespace LeftMerge86914
def owner : Owner := ⟨.program ⟨214⟩, ⟨25298⟩⟩
def mergeEvent : Nat := 86914
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86909RawTerms
def rightRaw : List Term := Proof.Events338.exact86725RawTerms
def group : MergeGroup := .operator 86909 86725
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86909) (leftOrdinal := 1)
    (rightResult := 86725) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86914

namespace LeftMerge86922
def owner : Owner := ⟨.program ⟨214⟩, ⟨27000⟩⟩
def mergeEvent : Nat := 86922
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26998⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86916RawTerms
def rightRaw : List Term := Proof.Events338.exact86641RawTerms
def group : MergeGroup := .operator 86916 86641
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86916) (leftOrdinal := 0)
    (rightResult := 86641) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26998⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26998⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86922

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
