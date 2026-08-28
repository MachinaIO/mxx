import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge10752
def owner : Owner := ⟨.program ⟨214⟩, ⟨26242⟩⟩
def mergeEvent : Nat := 10752
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11653⟩⟩, ⟨.program ⟨214⟩, ⟨14677⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23676⟩⟩] } }
def leftRaw : List Term := Proof.Events041.exact10748RawTerms
def rightRaw : List Term := Proof.Events041.exact10562RawTerms
def group : MergeGroup := .operator 10748 10562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10748) (leftOrdinal := 2)
    (rightResult := 10562) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11653⟩⟩, ⟨.program ⟨214⟩, ⟨14677⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23676⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11653⟩⟩, ⟨.program ⟨214⟩, ⟨14677⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23676⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11653⟩⟩, ⟨.program ⟨214⟩, ⟨14677⟩⟩], [⟨.program ⟨214⟩, ⟨23676⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10752

namespace LeftMerge10753
def owner : Owner := ⟨.program ⟨214⟩, ⟨26242⟩⟩
def mergeEvent : Nat := 10753
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26240⟩⟩] } }
def leftRaw : List Term := Proof.Events041.exact10748RawTerms
def rightRaw : List Term := Proof.Events041.exact10562RawTerms
def group : MergeGroup := .operator 10748 10562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10748) (leftOrdinal := 1)
    (rightResult := 10562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26240⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26240⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26240⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10753

namespace LeftMerge10761
def owner : Owner := ⟨.program ⟨214⟩, ⟨28354⟩⟩
def mergeEvent : Nat := 10761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10755RawTerms
def rightRaw : List Term := Proof.Events040.exact10459RawTerms
def group : MergeGroup := .operator 10755 10459
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10755) (leftOrdinal := 1)
    (rightResult := 10459) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28352⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10761

namespace LeftMerge10763
def owner : Owner := ⟨.program ⟨214⟩, ⟨28354⟩⟩
def mergeEvent : Nat := 10763
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24300⟩⟩] } }
def rhsRaw : List Term := Proof.Events040.exact10456RawTerms
def group : MergeGroup := .relation 10762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 10762) (rhsResult := 10456)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28352⟩⟩) ⟨24300⟩ 10456) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24300⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10763

namespace LeftMerge10764
def owner : Owner := ⟨.program ⟨214⟩, ⟨28354⟩⟩
def mergeEvent : Nat := 10764
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10755RawTerms
def rightRaw : List Term := Proof.Events040.exact10459RawTerms
def group : MergeGroup := .operator 10755 10459
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10755) (leftOrdinal := 0)
    (rightResult := 10459) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28352⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10764

namespace LeftMerge10778
def owner : Owner := ⟨.program ⟨214⟩, ⟨21707⟩⟩
def mergeEvent : Nat := 10778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events042.exact10772RawTerms
def group : MergeGroup := .operator 6561 10772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 10772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21704⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10778

namespace LeftMerge10899
def owner : Owner := ⟨.program ⟨214⟩, ⟨16236⟩⟩
def mergeEvent : Nat := 10899
def frameStart : Nat := 10833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10895RawTerms
def rightRaw : List Term := Proof.Events042.exact10893RawTerms
def group : MergeGroup := .operator 10895 10893
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10895) (leftOrdinal := 0)
    (rightResult := 10893) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10899

namespace LeftMerge10911
def owner : Owner := ⟨.program ⟨214⟩, ⟨28353⟩⟩
def mergeEvent : Nat := 10911
def frameStart : Nat := 10833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10907RawTerms
def rightRaw : List Term := Proof.Events042.exact10884RawTerms
def group : MergeGroup := .operator 10907 10884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10907) (leftOrdinal := 1)
    (rightResult := 10884) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28352⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10911

namespace LeftMerge10913
def owner : Owner := ⟨.program ⟨214⟩, ⟨28353⟩⟩
def mergeEvent : Nat := 10913
def frameStart : Nat := 10833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24300⟩⟩] } }
def rhsRaw : List Term := Proof.Events042.exact10881RawTerms
def group : MergeGroup := .relation 10912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 10912) (rhsResult := 10881)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28352⟩⟩) ⟨24300⟩ 10881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24300⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10913

namespace LeftMerge10914
def owner : Owner := ⟨.program ⟨214⟩, ⟨28353⟩⟩
def mergeEvent : Nat := 10914
def frameStart : Nat := 10833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10907RawTerms
def rightRaw : List Term := Proof.Events042.exact10884RawTerms
def group : MergeGroup := .operator 10907 10884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10907) (leftOrdinal := 0)
    (rightResult := 10884) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28352⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10914

namespace LeftMerge10922
def owner : Owner := ⟨.program ⟨214⟩, ⟨18403⟩⟩
def mergeEvent : Nat := 10922
def frameStart : Nat := 10833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10895RawTerms
def rightRaw : List Term := Proof.Events042.exact10918RawTerms
def group : MergeGroup := .operator 10895 10918
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10895) (leftOrdinal := 0)
    (rightResult := 10918) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10922

namespace LeftMerge10939
def owner : Owner := ⟨.program ⟨214⟩, ⟨21707⟩⟩
def mergeEvent : Nat := 10939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24300⟩⟩] } }
def rhsRaw : List Term := Proof.Events042.exact10936RawTerms
def group : MergeGroup := .relation 10938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 10938) (rhsResult := 10936)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 10937 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩) (none) 10936) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24300⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10939

namespace LeftMerge10940
def owner : Owner := ⟨.program ⟨214⟩, ⟨21707⟩⟩
def mergeEvent : Nat := 10940
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩] } }
def rhsRaw : List Term := Proof.Events042.exact10936RawTerms
def group : MergeGroup := .relation 10938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 10938) (rhsResult := 10936)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 10937 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩) (none) 10936) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10940

namespace LeftMerge10941
def owner : Owner := ⟨.program ⟨214⟩, ⟨21707⟩⟩
def mergeEvent : Nat := 10941
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events042.exact10936RawTerms
def group : MergeGroup := .relation 10938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 10938) (rhsResult := 10936)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 10937 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩) (none) 10936) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10941

namespace LeftMerge10942
def owner : Owner := ⟨.program ⟨214⟩, ⟨21707⟩⟩
def mergeEvent : Nat := 10942
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }
def rhsRaw : List Term := Proof.Events042.exact10936RawTerms
def group : MergeGroup := .relation 10938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 10938) (rhsResult := 10936)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 10937 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩) (none) 10936) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10942

namespace LeftMerge10947
def owner : Owner := ⟨.program ⟨214⟩, ⟨28355⟩⟩
def mergeEvent : Nat := 10947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24300⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10943RawTerms
def rightRaw : List Term := Proof.Events042.exact10765RawTerms
def group : MergeGroup := .operator 10943 10765
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10943) (leftOrdinal := 2)
    (rightResult := 10765) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24300⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24300⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10947

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
