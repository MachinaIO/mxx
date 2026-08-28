import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge73773
def owner : Owner := ⟨.program ⟨214⟩, ⟨26348⟩⟩
def mergeEvent : Nat := 73773
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73767RawTerms
def rightRaw : List Term := Proof.Events287.exact73490RawTerms
def group : MergeGroup := .operator 73767 73490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73767) (leftOrdinal := 0)
    (rightResult := 73490) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26346⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73773

namespace LeftMerge73774
def owner : Owner := ⟨.program ⟨214⟩, ⟨26348⟩⟩
def mergeEvent : Nat := 73774
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73767RawTerms
def rightRaw : List Term := Proof.Events287.exact73490RawTerms
def group : MergeGroup := .operator 73767 73490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73767) (leftOrdinal := 1)
    (rightResult := 73490) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26346⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73774

namespace LeftMerge73776
def owner : Owner := ⟨.program ⟨214⟩, ⟨26348⟩⟩
def mergeEvent : Nat := 73776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23718⟩⟩] } }
def rhsRaw : List Term := Proof.Events287.exact73487RawTerms
def group : MergeGroup := .relation 73775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73775) (rhsResult := 73487)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26346⟩⟩) ⟨23718⟩ 73487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23718⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73776

namespace LeftMerge73790
def owner : Owner := ⟨.program ⟨214⟩, ⟨20391⟩⟩
def mergeEvent : Nat := 73790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events288.exact73784RawTerms
def group : MergeGroup := .operator 65387 73784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 73784) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20388⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73790

namespace LeftMerge73911
def owner : Owner := ⟨.program ⟨214⟩, ⟨14830⟩⟩
def mergeEvent : Nat := 73911
def frameStart : Nat := 73845
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73907RawTerms
def rightRaw : List Term := Proof.Events288.exact73905RawTerms
def group : MergeGroup := .operator 73907 73905
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73907) (leftOrdinal := 0)
    (rightResult := 73905) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73911

namespace LeftMerge73923
def owner : Owner := ⟨.program ⟨214⟩, ⟨26347⟩⟩
def mergeEvent : Nat := 73923
def frameStart : Nat := 73845
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73919RawTerms
def rightRaw : List Term := Proof.Events288.exact73896RawTerms
def group : MergeGroup := .operator 73919 73896
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73919) (leftOrdinal := 0)
    (rightResult := 73896) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26346⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73923

namespace LeftMerge73924
def owner : Owner := ⟨.program ⟨214⟩, ⟨26347⟩⟩
def mergeEvent : Nat := 73924
def frameStart : Nat := 73845
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73919RawTerms
def rightRaw : List Term := Proof.Events288.exact73896RawTerms
def group : MergeGroup := .operator 73919 73896
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73919) (leftOrdinal := 1)
    (rightResult := 73896) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26346⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73924

namespace LeftMerge73926
def owner : Owner := ⟨.program ⟨214⟩, ⟨26347⟩⟩
def mergeEvent : Nat := 73926
def frameStart : Nat := 73845
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23718⟩⟩] } }
def rhsRaw : List Term := Proof.Events288.exact73893RawTerms
def group : MergeGroup := .relation 73925
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73925) (rhsResult := 73893)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26346⟩⟩) ⟨23718⟩ 73893) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23718⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73926

namespace LeftMerge73934
def owner : Owner := ⟨.program ⟨214⟩, ⟨15263⟩⟩
def mergeEvent : Nat := 73934
def frameStart : Nat := 73845
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73907RawTerms
def rightRaw : List Term := Proof.Events288.exact73930RawTerms
def group : MergeGroup := .operator 73907 73930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73907) (leftOrdinal := 0)
    (rightResult := 73930) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73934

namespace LeftMerge73951
def owner : Owner := ⟨.program ⟨214⟩, ⟨20391⟩⟩
def mergeEvent : Nat := 73951
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }
def rhsRaw : List Term := Proof.Events288.exact73948RawTerms
def group : MergeGroup := .relation 73950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73950) (rhsResult := 73948)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73949 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩) (none) 73948) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73951

namespace LeftMerge73952
def owner : Owner := ⟨.program ⟨214⟩, ⟨20391⟩⟩
def mergeEvent : Nat := 73952
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }
def rhsRaw : List Term := Proof.Events288.exact73948RawTerms
def group : MergeGroup := .relation 73950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73950) (rhsResult := 73948)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73949 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩) (none) 73948) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73952

namespace LeftMerge73953
def owner : Owner := ⟨.program ⟨214⟩, ⟨20391⟩⟩
def mergeEvent : Nat := 73953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23718⟩⟩] } }
def rhsRaw : List Term := Proof.Events288.exact73948RawTerms
def group : MergeGroup := .relation 73950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73950) (rhsResult := 73948)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73949 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩) (none) 73948) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23718⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73953

namespace LeftMerge73954
def owner : Owner := ⟨.program ⟨214⟩, ⟨20391⟩⟩
def mergeEvent : Nat := 73954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events288.exact73948RawTerms
def group : MergeGroup := .relation 73950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73950) (rhsResult := 73948)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73949 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩) (none) 73948) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73954

namespace LeftMerge73959
def owner : Owner := ⟨.program ⟨214⟩, ⟨26349⟩⟩
def mergeEvent : Nat := 73959
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73955RawTerms
def rightRaw : List Term := Proof.Events288.exact73777RawTerms
def group : MergeGroup := .operator 73955 73777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73955) (leftOrdinal := 0)
    (rightResult := 73777) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73959

namespace LeftMerge73960
def owner : Owner := ⟨.program ⟨214⟩, ⟨26349⟩⟩
def mergeEvent : Nat := 73960
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23718⟩⟩] } }
def leftRaw : List Term := Proof.Events288.exact73955RawTerms
def rightRaw : List Term := Proof.Events288.exact73777RawTerms
def group : MergeGroup := .operator 73955 73777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73955) (leftOrdinal := 2)
    (rightResult := 73777) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23718⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23718⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73960

namespace LeftMerge74053
def owner : Owner := ⟨.program ⟨214⟩, ⟨30100⟩⟩
def mergeEvent : Nat := 74053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events289.exact74047RawTerms
def rightRaw : List Term := Proof.Events254.exact65270RawTerms
def group : MergeGroup := .operator 74047 65270
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74047) (leftOrdinal := 17)
    (rightResult := 65270) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge74053

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
