import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge39890
def owner : Owner := ⟨.program ⟨214⟩, ⟨28546⟩⟩
def mergeEvent : Nat := 39890
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39885RawTerms
def rightRaw : List Term := Proof.Events155.exact39707RawTerms
def group : MergeGroup := .operator 39885 39707
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39885) (leftOrdinal := 2)
    (rightResult := 39707) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24357⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39890

namespace LeftMerge39916
def owner : Owner := ⟨.program ⟨214⟩, ⟨11646⟩⟩
def mergeEvent : Nat := 39916
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events006.exact1774RawTerms
def rightRaw : List Term := Proof.Events140.exact36045RawTerms
def group : MergeGroup := .operator 1774 36045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1774) (leftOrdinal := 0)
    (rightResult := 36045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11645⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39916

namespace LeftMerge39921
def owner : Owner := ⟨.program ⟨214⟩, ⟨7313⟩⟩
def mergeEvent : Nat := 39921
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35915RawTerms
def rightRaw : List Term := Proof.Events040.exact10480RawTerms
def group : MergeGroup := .operator 35915 10480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35915) (leftOrdinal := 0)
    (rightResult := 10480) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39921

namespace LeftMerge39938
def owner : Owner := ⟨.program ⟨214⟩, ⟨14662⟩⟩
def mergeEvent : Nat := 39938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39932RawTerms
def rightRaw : List Term := Proof.Events006.exact1777RawTerms
def group : MergeGroup := .operator 39932 1777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39932) (leftOrdinal := 1)
    (rightResult := 1777) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39938

namespace LeftMerge39939
def owner : Owner := ⟨.program ⟨214⟩, ⟨14662⟩⟩
def mergeEvent : Nat := 39939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39932RawTerms
def rightRaw : List Term := Proof.Events006.exact1777RawTerms
def group : MergeGroup := .operator 39932 1777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39932) (leftOrdinal := 0)
    (rightResult := 1777) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39939

namespace LeftMerge39944
def owner : Owner := ⟨.program ⟨214⟩, ⟨14663⟩⟩
def mergeEvent : Nat := 39944
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events006.exact1777RawTerms
def rightRaw : List Term := Proof.Events140.exact36045RawTerms
def group : MergeGroup := .operator 1777 36045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1777) (leftOrdinal := 0)
    (rightResult := 36045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39944

namespace LeftMerge39949
def owner : Owner := ⟨.program ⟨214⟩, ⟨7294⟩⟩
def mergeEvent : Nat := 39949
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35915RawTerms
def rightRaw : List Term := Proof.Events041.exact10521RawTerms
def group : MergeGroup := .operator 35915 10521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35915) (leftOrdinal := 0)
    (rightResult := 10521) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39949

namespace LeftMerge39966
def owner : Owner := ⟨.program ⟨214⟩, ⟨14666⟩⟩
def mergeEvent : Nat := 39966
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact39960RawTerms
def rightRaw : List Term := Proof.Events041.exact10510RawTerms
def group : MergeGroup := .operator 39960 10510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39960) (leftOrdinal := 1)
    (rightResult := 10510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7858⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39966

namespace LeftMerge39968
def owner : Owner := ⟨.program ⟨214⟩, ⟨14666⟩⟩
def mergeEvent : Nat := 39968
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }
def rhsRaw : List Term := Proof.Events040.exact10480RawTerms
def group : MergeGroup := .relation 39967
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39967) (rhsResult := 10480)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7858⟩⟩) ⟨6781⟩ 10480) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39968

namespace LeftMerge39969
def owner : Owner := ⟨.program ⟨214⟩, ⟨14666⟩⟩
def mergeEvent : Nat := 39969
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact39960RawTerms
def rightRaw : List Term := Proof.Events041.exact10510RawTerms
def group : MergeGroup := .operator 39960 10510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39960) (leftOrdinal := 0)
    (rightResult := 10510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7858⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39969

namespace LeftMerge39974
def owner : Owner := ⟨.program ⟨214⟩, ⟨14667⟩⟩
def mergeEvent : Nat := 39974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact39970RawTerms
def rightRaw : List Term := Proof.Events156.exact39940RawTerms
def group : MergeGroup := .operator 39970 39940
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39970) (leftOrdinal := 1)
    (rightResult := 39940) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39974

namespace LeftMerge39982
def owner : Owner := ⟨.program ⟨214⟩, ⟨26231⟩⟩
def mergeEvent : Nat := 39982
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact39976RawTerms
def rightRaw : List Term := Proof.Events155.exact39912RawTerms
def group : MergeGroup := .operator 39976 39912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39976) (leftOrdinal := 1)
    (rightResult := 39912) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26230⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39982

namespace LeftMerge39984
def owner : Owner := ⟨.program ⟨214⟩, ⟨26231⟩⟩
def mergeEvent : Nat := 39984
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23672⟩⟩] } }
def rhsRaw : List Term := Proof.Events155.exact39909RawTerms
def group : MergeGroup := .relation 39983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39983) (rhsResult := 39909)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26230⟩⟩) ⟨23672⟩ 39909) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23672⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨23672⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39984

namespace LeftMerge39985
def owner : Owner := ⟨.program ⟨214⟩, ⟨26231⟩⟩
def mergeEvent : Nat := 39985
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact39976RawTerms
def rightRaw : List Term := Proof.Events155.exact39912RawTerms
def group : MergeGroup := .operator 39976 39912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39976) (leftOrdinal := 0)
    (rightResult := 39912) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26230⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39985

namespace LeftMerge39999
def owner : Owner := ⟨.program ⟨214⟩, ⟨19683⟩⟩
def mergeEvent : Nat := 39999
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events156.exact39993RawTerms
def group : MergeGroup := .operator 36137 39993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 39993) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19680⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39999

namespace LeftMerge40078
def owner : Owner := ⟨.program ⟨214⟩, ⟨14660⟩⟩
def mergeEvent : Nat := 40078
def frameStart : Nat := 40048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events156.exact40074RawTerms
def rightRaw : List Term := Proof.Events156.exact40071RawTerms
def group : MergeGroup := .operator 40074 40071
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40074) (leftOrdinal := 0)
    (rightResult := 40071) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11645⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40078

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
