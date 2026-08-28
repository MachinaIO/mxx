import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge72947
def owner : Owner := ⟨.program ⟨214⟩, ⟨15152⟩⟩
def mergeEvent : Nat := 72947
def frameStart : Nat := 72881
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72943RawTerms
def rightRaw : List Term := Proof.Events284.exact72941RawTerms
def group : MergeGroup := .operator 72943 72941
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72943) (leftOrdinal := 0)
    (rightResult := 72941) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72947

namespace LeftMerge72959
def owner : Owner := ⟨.program ⟨214⟩, ⟨26769⟩⟩
def mergeEvent : Nat := 72959
def frameStart : Nat := 72881
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72955RawTerms
def rightRaw : List Term := Proof.Events284.exact72932RawTerms
def group : MergeGroup := .operator 72955 72932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72955) (leftOrdinal := 0)
    (rightResult := 72932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26768⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72959

namespace LeftMerge72960
def owner : Owner := ⟨.program ⟨214⟩, ⟨26769⟩⟩
def mergeEvent : Nat := 72960
def frameStart : Nat := 72881
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72955RawTerms
def rightRaw : List Term := Proof.Events284.exact72932RawTerms
def group : MergeGroup := .operator 72955 72932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72955) (leftOrdinal := 1)
    (rightResult := 72932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26768⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72960

namespace LeftMerge72962
def owner : Owner := ⟨.program ⟨214⟩, ⟨26769⟩⟩
def mergeEvent : Nat := 72962
def frameStart : Nat := 72881
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23844⟩⟩] } }
def rhsRaw : List Term := Proof.Events284.exact72929RawTerms
def group : MergeGroup := .relation 72961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72961) (rhsResult := 72929)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26768⟩⟩) ⟨23844⟩ 72929) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23844⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72962

namespace LeftMerge72970
def owner : Owner := ⟨.program ⟨214⟩, ⟨15364⟩⟩
def mergeEvent : Nat := 72970
def frameStart : Nat := 72881
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events284.exact72943RawTerms
def rightRaw : List Term := Proof.Events285.exact72966RawTerms
def group : MergeGroup := .operator 72943 72966
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72943) (leftOrdinal := 0)
    (rightResult := 72966) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72970

namespace LeftMerge72987
def owner : Owner := ⟨.program ⟨214⟩, ⟨20679⟩⟩
def mergeEvent : Nat := 72987
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }
def rhsRaw : List Term := Proof.Events285.exact72984RawTerms
def group : MergeGroup := .relation 72986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72986) (rhsResult := 72984)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 72985 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩) (none) 72984) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72987

namespace LeftMerge72988
def owner : Owner := ⟨.program ⟨214⟩, ⟨20679⟩⟩
def mergeEvent : Nat := 72988
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩] } }
def rhsRaw : List Term := Proof.Events285.exact72984RawTerms
def group : MergeGroup := .relation 72986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72986) (rhsResult := 72984)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 72985 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩) (none) 72984) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72988

namespace LeftMerge72989
def owner : Owner := ⟨.program ⟨214⟩, ⟨20679⟩⟩
def mergeEvent : Nat := 72989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23844⟩⟩] } }
def rhsRaw : List Term := Proof.Events285.exact72984RawTerms
def group : MergeGroup := .relation 72986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72986) (rhsResult := 72984)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 72985 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩) (none) 72984) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23844⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72989

namespace LeftMerge72990
def owner : Owner := ⟨.program ⟨214⟩, ⟨20679⟩⟩
def mergeEvent : Nat := 72990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events285.exact72984RawTerms
def group : MergeGroup := .relation 72986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 72986) (rhsResult := 72984)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 72985 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩) (none) 72984) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72990

namespace LeftMerge72995
def owner : Owner := ⟨.program ⟨214⟩, ⟨26771⟩⟩
def mergeEvent : Nat := 72995
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact72991RawTerms
def rightRaw : List Term := Proof.Events284.exact72813RawTerms
def group : MergeGroup := .operator 72991 72813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72991) (leftOrdinal := 0)
    (rightResult := 72813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge72995

namespace LeftMerge72996
def owner : Owner := ⟨.program ⟨214⟩, ⟨26771⟩⟩
def mergeEvent : Nat := 72996
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23844⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact72991RawTerms
def rightRaw : List Term := Proof.Events284.exact72813RawTerms
def group : MergeGroup := .operator 72991 72813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 72991) (leftOrdinal := 2)
    (rightResult := 72813) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23844⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23844⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge72996

namespace LeftMerge73022
def owner : Owner := ⟨.program ⟨214⟩, ⟨10671⟩⟩
def mergeEvent : Nat := 73022
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events013.exact3454RawTerms
def rightRaw : List Term := Proof.Events255.exact65295RawTerms
def group : MergeGroup := .operator 3454 65295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3454) (leftOrdinal := 0)
    (rightResult := 65295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73022

namespace LeftMerge73027
def owner : Owner := ⟨.program ⟨214⟩, ⟨7191⟩⟩
def mergeEvent : Nat := 73027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65165RawTerms
def rightRaw : List Term := Proof.Events056.exact14488RawTerms
def group : MergeGroup := .operator 65165 14488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65165) (leftOrdinal := 0)
    (rightResult := 14488) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73027

namespace LeftMerge73044
def owner : Owner := ⟨.program ⟨214⟩, ⟨10674⟩⟩
def mergeEvent : Nat := 73044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact73038RawTerms
def rightRaw : List Term := Proof.Events013.exact3457RawTerms
def group : MergeGroup := .operator 73038 3457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73038) (leftOrdinal := 1)
    (rightResult := 3457) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73044

namespace LeftMerge73045
def owner : Owner := ⟨.program ⟨214⟩, ⟨10674⟩⟩
def mergeEvent : Nat := 73045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact73038RawTerms
def rightRaw : List Term := Proof.Events013.exact3457RawTerms
def group : MergeGroup := .operator 73038 3457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73038) (leftOrdinal := 0)
    (rightResult := 3457) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73045

namespace LeftMerge73050
def owner : Owner := ⟨.program ⟨214⟩, ⟨9501⟩⟩
def mergeEvent : Nat := 73050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events013.exact3457RawTerms
def rightRaw : List Term := Proof.Events255.exact65295RawTerms
def group : MergeGroup := .operator 3457 65295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3457) (leftOrdinal := 0)
    (rightResult := 65295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73050

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
