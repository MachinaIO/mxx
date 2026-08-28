import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge25795
def owner : Owner := ⟨.program ⟨214⟩, ⟨14454⟩⟩
def mergeEvent : Nat := 25795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25789RawTerms
def rightRaw : List Term := Proof.Events004.exact1052RawTerms
def group : MergeGroup := .operator 25789 1052
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25789) (leftOrdinal := 1)
    (rightResult := 1052) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25795

namespace LeftMerge25796
def owner : Owner := ⟨.program ⟨214⟩, ⟨14454⟩⟩
def mergeEvent : Nat := 25796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25789RawTerms
def rightRaw : List Term := Proof.Events004.exact1052RawTerms
def group : MergeGroup := .operator 25789 1052
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25789) (leftOrdinal := 0)
    (rightResult := 1052) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25796

namespace LeftMerge25801
def owner : Owner := ⟨.program ⟨214⟩, ⟨14455⟩⟩
def mergeEvent : Nat := 25801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events004.exact1052RawTerms
def rightRaw : List Term := Proof.Events083.exact21420RawTerms
def group : MergeGroup := .operator 1052 21420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1052) (leftOrdinal := 0)
    (rightResult := 21420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25801

namespace LeftMerge25806
def owner : Owner := ⟨.program ⟨214⟩, ⟨7331⟩⟩
def mergeEvent : Nat := 25806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21290RawTerms
def rightRaw : List Term := Proof.Events043.exact11022RawTerms
def group : MergeGroup := .operator 21290 11022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21290) (leftOrdinal := 0)
    (rightResult := 11022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25806

namespace LeftMerge25823
def owner : Owner := ⟨.program ⟨214⟩, ⟨14458⟩⟩
def mergeEvent : Nat := 25823
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25817RawTerms
def rightRaw : List Term := Proof.Events043.exact11011RawTerms
def group : MergeGroup := .operator 25817 11011
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25817) (leftOrdinal := 1)
    (rightResult := 11011) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7855⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25823

namespace LeftMerge25825
def owner : Owner := ⟨.program ⟨214⟩, ⟨14458⟩⟩
def mergeEvent : Nat := 25825
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def rhsRaw : List Term := Proof.Events042.exact10981RawTerms
def group : MergeGroup := .relation 25824
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25824) (rhsResult := 10981)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7855⟩⟩) ⟨6780⟩ 10981) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25825

namespace LeftMerge25826
def owner : Owner := ⟨.program ⟨214⟩, ⟨14458⟩⟩
def mergeEvent : Nat := 25826
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25817RawTerms
def rightRaw : List Term := Proof.Events043.exact11011RawTerms
def group : MergeGroup := .operator 25817 11011
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25817) (leftOrdinal := 0)
    (rightResult := 11011) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7855⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25826

namespace LeftMerge25831
def owner : Owner := ⟨.program ⟨214⟩, ⟨14459⟩⟩
def mergeEvent : Nat := 25831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25827RawTerms
def rightRaw : List Term := Proof.Events100.exact25797RawTerms
def group : MergeGroup := .operator 25827 25797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25827) (leftOrdinal := 1)
    (rightResult := 25797) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25831

namespace LeftMerge25839
def owner : Owner := ⟨.program ⟨214⟩, ⟨26159⟩⟩
def mergeEvent : Nat := 25839
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25833RawTerms
def rightRaw : List Term := Proof.Events100.exact25769RawTerms
def group : MergeGroup := .operator 25833 25769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25833) (leftOrdinal := 1)
    (rightResult := 25769) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26158⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25839

namespace LeftMerge25841
def owner : Owner := ⟨.program ⟨214⟩, ⟨26159⟩⟩
def mergeEvent : Nat := 25841
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23632⟩⟩] } }
def rhsRaw : List Term := Proof.Events100.exact25766RawTerms
def group : MergeGroup := .relation 25840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25840) (rhsResult := 25766)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26158⟩⟩) ⟨23632⟩ 25766) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23632⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25841

namespace LeftMerge25842
def owner : Owner := ⟨.program ⟨214⟩, ⟨26159⟩⟩
def mergeEvent : Nat := 25842
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25833RawTerms
def rightRaw : List Term := Proof.Events100.exact25769RawTerms
def group : MergeGroup := .operator 25833 25769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25833) (leftOrdinal := 0)
    (rightResult := 25769) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26158⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25842

namespace LeftMerge25856
def owner : Owner := ⟨.program ⟨214⟩, ⟨19615⟩⟩
def mergeEvent : Nat := 25856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19612⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events100.exact25850RawTerms
def group : MergeGroup := .operator 21512 25850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 25850) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19612⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25856

namespace LeftMerge25935
def owner : Owner := ⟨.program ⟨214⟩, ⟨14452⟩⟩
def mergeEvent : Nat := 25935
def frameStart : Nat := 25905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events101.exact25931RawTerms
def rightRaw : List Term := Proof.Events101.exact25928RawTerms
def group : MergeGroup := .operator 25931 25928
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25931) (leftOrdinal := 0)
    (rightResult := 25928) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11565⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25935

namespace LeftMerge25965
def owner : Owner := ⟨.program ⟨214⟩, ⟨14545⟩⟩
def mergeEvent : Nat := 25965
def frameStart : Nat := 25905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact25961RawTerms
def rightRaw : List Term := Proof.Events101.exact25959RawTerms
def group : MergeGroup := .operator 25961 25959
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25961) (leftOrdinal := 0)
    (rightResult := 25959) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25965

namespace LeftMerge25988
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def mergeEvent : Nat := 25988
def frameStart : Nat := 25905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact25984RawTerms
def rightRaw : List Term := Proof.Events101.exact25981RawTerms
def group : MergeGroup := .operator 25984 25981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25984) (leftOrdinal := 0)
    (rightResult := 25981) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7855⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25988

namespace LeftMerge25997
def owner : Owner := ⟨.program ⟨214⟩, ⟨26161⟩⟩
def mergeEvent : Nat := 25997
def frameStart : Nat := 25905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact25993RawTerms
def rightRaw : List Term := Proof.Events101.exact25950RawTerms
def group : MergeGroup := .operator 25993 25950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25993) (leftOrdinal := 0)
    (rightResult := 25950) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26158⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25997

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
