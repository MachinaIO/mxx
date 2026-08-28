import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge39694
def owner : Owner := ⟨.program ⟨214⟩, ⟨25154⟩⟩
def mergeEvent : Nat := 39694
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23084⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39690RawTerms
def rightRaw : List Term := Proof.Events154.exact39504RawTerms
def group : MergeGroup := .operator 39690 39504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39690) (leftOrdinal := 2)
    (rightResult := 39504) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23084⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23084⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], [⟨.program ⟨214⟩, ⟨23084⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39694

namespace LeftMerge39695
def owner : Owner := ⟨.program ⟨214⟩, ⟨25154⟩⟩
def mergeEvent : Nat := 39695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25152⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39690RawTerms
def rightRaw : List Term := Proof.Events154.exact39504RawTerms
def group : MergeGroup := .operator 39690 39504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39690) (leftOrdinal := 1)
    (rightResult := 39504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25152⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25152⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25152⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39695

namespace LeftMerge39703
def owner : Owner := ⟨.program ⟨214⟩, ⟨28545⟩⟩
def mergeEvent : Nat := 39703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39697RawTerms
def rightRaw : List Term := Proof.Events153.exact39420RawTerms
def group : MergeGroup := .operator 39697 39420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39697) (leftOrdinal := 0)
    (rightResult := 39420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28543⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39703

namespace LeftMerge39704
def owner : Owner := ⟨.program ⟨214⟩, ⟨28545⟩⟩
def mergeEvent : Nat := 39704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39697RawTerms
def rightRaw : List Term := Proof.Events153.exact39420RawTerms
def group : MergeGroup := .operator 39697 39420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39697) (leftOrdinal := 1)
    (rightResult := 39420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28543⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39704

namespace LeftMerge39706
def owner : Owner := ⟨.program ⟨214⟩, ⟨28545⟩⟩
def mergeEvent : Nat := 39706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }
def rhsRaw : List Term := Proof.Events153.exact39417RawTerms
def group : MergeGroup := .relation 39705
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39705) (rhsResult := 39417)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28543⟩⟩) ⟨24357⟩ 39417) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24357⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39706

namespace LeftMerge39720
def owner : Owner := ⟨.program ⟨214⟩, ⟨21843⟩⟩
def mergeEvent : Nat := 39720
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events155.exact39714RawTerms
def group : MergeGroup := .operator 36137 39714
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 39714) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21840⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39720

namespace LeftMerge39841
def owner : Owner := ⟨.program ⟨214⟩, ⟨16347⟩⟩
def mergeEvent : Nat := 39841
def frameStart : Nat := 39775
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39837RawTerms
def rightRaw : List Term := Proof.Events155.exact39835RawTerms
def group : MergeGroup := .operator 39837 39835
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39837) (leftOrdinal := 0)
    (rightResult := 39835) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39841

namespace LeftMerge39853
def owner : Owner := ⟨.program ⟨214⟩, ⟨28544⟩⟩
def mergeEvent : Nat := 39853
def frameStart : Nat := 39775
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39849RawTerms
def rightRaw : List Term := Proof.Events155.exact39826RawTerms
def group : MergeGroup := .operator 39849 39826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39849) (leftOrdinal := 0)
    (rightResult := 39826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28543⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39853

namespace LeftMerge39854
def owner : Owner := ⟨.program ⟨214⟩, ⟨28544⟩⟩
def mergeEvent : Nat := 39854
def frameStart : Nat := 39775
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39849RawTerms
def rightRaw : List Term := Proof.Events155.exact39826RawTerms
def group : MergeGroup := .operator 39849 39826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39849) (leftOrdinal := 1)
    (rightResult := 39826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28543⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39854

namespace LeftMerge39856
def owner : Owner := ⟨.program ⟨214⟩, ⟨28544⟩⟩
def mergeEvent : Nat := 39856
def frameStart : Nat := 39775
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }
def rhsRaw : List Term := Proof.Events155.exact39823RawTerms
def group : MergeGroup := .relation 39855
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39855) (rhsResult := 39823)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28543⟩⟩) ⟨24357⟩ 39823) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24357⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39856

namespace LeftMerge39864
def owner : Owner := ⟨.program ⟨214⟩, ⟨16315⟩⟩
def mergeEvent : Nat := 39864
def frameStart : Nat := 39775
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39837RawTerms
def rightRaw : List Term := Proof.Events155.exact39860RawTerms
def group : MergeGroup := .operator 39837 39860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39837) (leftOrdinal := 0)
    (rightResult := 39860) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39864

namespace LeftMerge39881
def owner : Owner := ⟨.program ⟨214⟩, ⟨21843⟩⟩
def mergeEvent : Nat := 39881
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }
def rhsRaw : List Term := Proof.Events155.exact39878RawTerms
def group : MergeGroup := .relation 39880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39880) (rhsResult := 39878)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39879 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩) (none) 39878) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39881

namespace LeftMerge39882
def owner : Owner := ⟨.program ⟨214⟩, ⟨21843⟩⟩
def mergeEvent : Nat := 39882
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } }
def rhsRaw : List Term := Proof.Events155.exact39878RawTerms
def group : MergeGroup := .relation 39880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39880) (rhsResult := 39878)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39879 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩) (none) 39878) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39882

namespace LeftMerge39883
def owner : Owner := ⟨.program ⟨214⟩, ⟨21843⟩⟩
def mergeEvent : Nat := 39883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }
def rhsRaw : List Term := Proof.Events155.exact39878RawTerms
def group : MergeGroup := .relation 39880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39880) (rhsResult := 39878)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39879 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩) (none) 39878) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24357⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39883

namespace LeftMerge39884
def owner : Owner := ⟨.program ⟨214⟩, ⟨21843⟩⟩
def mergeEvent : Nat := 39884
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events155.exact39878RawTerms
def group : MergeGroup := .relation 39880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39880) (rhsResult := 39878)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 39879 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩) (none) 39878) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39884

namespace LeftMerge39889
def owner : Owner := ⟨.program ⟨214⟩, ⟨28546⟩⟩
def mergeEvent : Nat := 39889
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39885RawTerms
def rightRaw : List Term := Proof.Events155.exact39707RawTerms
def group : MergeGroup := .operator 39885 39707
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39885) (leftOrdinal := 0)
    (rightResult := 39707) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39889

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
