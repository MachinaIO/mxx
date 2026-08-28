import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge53830
def owner : Owner := ⟨.program ⟨214⟩, ⟨19823⟩⟩
def mergeEvent : Nat := 53830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }
def rhsRaw : List Term := Proof.Events210.exact53826RawTerms
def group : MergeGroup := .relation 53828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53828) (rhsResult := 53826)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53827 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩) (none) 53826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53830

namespace LeftMerge53831
def owner : Owner := ⟨.program ⟨214⟩, ⟨19823⟩⟩
def mergeEvent : Nat := 53831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23124⟩⟩] } }
def rhsRaw : List Term := Proof.Events210.exact53826RawTerms
def group : MergeGroup := .relation 53828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53828) (rhsResult := 53826)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53827 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩) (none) 53826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23124⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], [⟨.program ⟨214⟩, ⟨23124⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53831

namespace LeftMerge53832
def owner : Owner := ⟨.program ⟨214⟩, ⟨19823⟩⟩
def mergeEvent : Nat := 53832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events210.exact53826RawTerms
def group : MergeGroup := .relation 53828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53828) (rhsResult := 53826)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53827 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩) (none) 53826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53832

namespace LeftMerge53837
def owner : Owner := ⟨.program ⟨214⟩, ⟨25226⟩⟩
def mergeEvent : Nat := 53837
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23124⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53833RawTerms
def rightRaw : List Term := Proof.Events209.exact53647RawTerms
def group : MergeGroup := .operator 53833 53647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53833) (leftOrdinal := 2)
    (rightResult := 53647) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23124⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23124⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], [⟨.program ⟨214⟩, ⟨23124⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53837

namespace LeftMerge53838
def owner : Owner := ⟨.program ⟨214⟩, ⟨25226⟩⟩
def mergeEvent : Nat := 53838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53833RawTerms
def rightRaw : List Term := Proof.Events209.exact53647RawTerms
def group : MergeGroup := .operator 53833 53647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53833) (leftOrdinal := 1)
    (rightResult := 53647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53838

namespace LeftMerge53846
def owner : Owner := ⟨.program ⟨214⟩, ⟨28749⟩⟩
def mergeEvent : Nat := 53846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53840RawTerms
def rightRaw : List Term := Proof.Events209.exact53563RawTerms
def group : MergeGroup := .operator 53840 53563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53840) (leftOrdinal := 0)
    (rightResult := 53563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28747⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53846

namespace LeftMerge53847
def owner : Owner := ⟨.program ⟨214⟩, ⟨28749⟩⟩
def mergeEvent : Nat := 53847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53840RawTerms
def rightRaw : List Term := Proof.Events209.exact53563RawTerms
def group : MergeGroup := .operator 53840 53563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53840) (leftOrdinal := 1)
    (rightResult := 53563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28747⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53847

namespace LeftMerge53849
def owner : Owner := ⟨.program ⟨214⟩, ⟨28749⟩⟩
def mergeEvent : Nat := 53849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24417⟩⟩] } }
def rhsRaw : List Term := Proof.Events209.exact53560RawTerms
def group : MergeGroup := .relation 53848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53848) (rhsResult := 53560)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28747⟩⟩) ⟨24417⟩ 53560) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24417⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨24417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53849

namespace LeftMerge53863
def owner : Owner := ⟨.program ⟨214⟩, ⟨21983⟩⟩
def mergeEvent : Nat := 53863
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21980⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events210.exact53857RawTerms
def group : MergeGroup := .operator 50762 53857
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 53857) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21980⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21980⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53863

namespace LeftMerge53984
def owner : Owner := ⟨.program ⟨214⟩, ⟨16427⟩⟩
def mergeEvent : Nat := 53984
def frameStart : Nat := 53918
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53980RawTerms
def rightRaw : List Term := Proof.Events210.exact53978RawTerms
def group : MergeGroup := .operator 53980 53978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53980) (leftOrdinal := 0)
    (rightResult := 53978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53984

namespace LeftMerge53996
def owner : Owner := ⟨.program ⟨214⟩, ⟨28748⟩⟩
def mergeEvent : Nat := 53996
def frameStart : Nat := 53918
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53992RawTerms
def rightRaw : List Term := Proof.Events210.exact53969RawTerms
def group : MergeGroup := .operator 53992 53969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53992) (leftOrdinal := 0)
    (rightResult := 53969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28747⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53996

namespace LeftMerge53997
def owner : Owner := ⟨.program ⟨214⟩, ⟨28748⟩⟩
def mergeEvent : Nat := 53997
def frameStart : Nat := 53918
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53992RawTerms
def rightRaw : List Term := Proof.Events210.exact53969RawTerms
def group : MergeGroup := .operator 53992 53969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53992) (leftOrdinal := 1)
    (rightResult := 53969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28747⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53997

namespace LeftMerge53999
def owner : Owner := ⟨.program ⟨214⟩, ⟨28748⟩⟩
def mergeEvent : Nat := 53999
def frameStart : Nat := 53918
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24417⟩⟩] } }
def rhsRaw : List Term := Proof.Events210.exact53966RawTerms
def group : MergeGroup := .relation 53998
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53998) (rhsResult := 53966)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28747⟩⟩) ⟨24417⟩ 53966) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24417⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16385⟩⟩], [⟨.program ⟨214⟩, ⟨24417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53999

namespace LeftMerge54007
def owner : Owner := ⟨.program ⟨214⟩, ⟨17124⟩⟩
def mergeEvent : Nat := 54007
def frameStart : Nat := 53918
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17123⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events210.exact53980RawTerms
def rightRaw : List Term := Proof.Events210.exact54003RawTerms
def group : MergeGroup := .operator 53980 54003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53980) (leftOrdinal := 0)
    (rightResult := 54003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17123⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54007

namespace LeftMerge54024
def owner : Owner := ⟨.program ⟨214⟩, ⟨21983⟩⟩
def mergeEvent : Nat := 54024
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }
def rhsRaw : List Term := Proof.Events211.exact54021RawTerms
def group : MergeGroup := .relation 54023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54023) (rhsResult := 54021)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21980⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54022 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21980⟩⟩]⟩) (none) 54021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54024

namespace LeftMerge54025
def owner : Owner := ⟨.program ⟨214⟩, ⟨21983⟩⟩
def mergeEvent : Nat := 54025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩] } }
def rhsRaw : List Term := Proof.Events211.exact54021RawTerms
def group : MergeGroup := .relation 54023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54023) (rhsResult := 54021)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21980⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54022 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21980⟩⟩]⟩) (none) 54021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54025

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
