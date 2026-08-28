import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge8712
def owner : Owner := ⟨.program ⟨214⟩, ⟨25473⟩⟩
def mergeEvent : Nat := 8712
def frameStart : Nat := 8620
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩] } }
def leftRaw : List Term := Proof.Events034.exact8708RawTerms
def rightRaw : List Term := Proof.Events033.exact8665RawTerms
def group : MergeGroup := .operator 8708 8665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8708) (leftOrdinal := 1)
    (rightResult := 8665) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25470⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8712

namespace LeftMerge8714
def owner : Owner := ⟨.program ⟨214⟩, ⟨25473⟩⟩
def mergeEvent : Nat := 8714
def frameStart : Nat := 8620
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23256⟩⟩] } }
def rhsRaw : List Term := Proof.Events033.exact8662RawTerms
def group : MergeGroup := .relation 8713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 8713) (rhsResult := 8662)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25470⟩⟩) ⟨23256⟩ 8662) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23256⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨23256⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8714

namespace LeftMerge8715
def owner : Owner := ⟨.program ⟨214⟩, ⟨25473⟩⟩
def mergeEvent : Nat := 8715
def frameStart : Nat := 8620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩] } }
def leftRaw : List Term := Proof.Events034.exact8708RawTerms
def rightRaw : List Term := Proof.Events033.exact8665RawTerms
def group : MergeGroup := .operator 8708 8665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8708) (leftOrdinal := 0)
    (rightResult := 8665) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25470⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8715

namespace LeftMerge8723
def owner : Owner := ⟨.program ⟨214⟩, ⟨16567⟩⟩
def mergeEvent : Nat := 8723
def frameStart : Nat := 8620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8676RawTerms
def rightRaw : List Term := Proof.Events034.exact8719RawTerms
def group : MergeGroup := .operator 8676 8719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8676) (leftOrdinal := 0)
    (rightResult := 8719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8723

namespace LeftMerge8740
def owner : Owner := ⟨.program ⟨214⟩, ⟨19979⟩⟩
def mergeEvent : Nat := 8740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23256⟩⟩] } }
def rhsRaw : List Term := Proof.Events034.exact8737RawTerms
def group : MergeGroup := .relation 8739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 8739) (rhsResult := 8737)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 8738 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩]⟩) (none) 8737) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23256⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨23256⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8740

namespace LeftMerge8741
def owner : Owner := ⟨.program ⟨214⟩, ⟨19979⟩⟩
def mergeEvent : Nat := 8741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩] } }
def rhsRaw : List Term := Proof.Events034.exact8737RawTerms
def group : MergeGroup := .relation 8739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 8739) (rhsResult := 8737)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 8738 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩]⟩) (none) 8737) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8741

namespace LeftMerge8742
def owner : Owner := ⟨.program ⟨214⟩, ⟨19979⟩⟩
def mergeEvent : Nat := 8742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events034.exact8737RawTerms
def group : MergeGroup := .relation 8739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 8739) (rhsResult := 8737)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 8738 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩]⟩) (none) 8737) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8742

namespace LeftMerge8743
def owner : Owner := ⟨.program ⟨214⟩, ⟨19979⟩⟩
def mergeEvent : Nat := 8743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }
def rhsRaw : List Term := Proof.Events034.exact8737RawTerms
def group : MergeGroup := .relation 8739
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 8739) (rhsResult := 8737)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 8738 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩]⟩) (none) 8737) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8743

namespace LeftMerge8748
def owner : Owner := ⟨.program ⟨214⟩, ⟨25472⟩⟩
def mergeEvent : Nat := 8748
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23256⟩⟩] } }
def leftRaw : List Term := Proof.Events034.exact8744RawTerms
def rightRaw : List Term := Proof.Events033.exact8558RawTerms
def group : MergeGroup := .operator 8744 8558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8744) (leftOrdinal := 2)
    (rightResult := 8558) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23256⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23256⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨23256⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8748

namespace LeftMerge8749
def owner : Owner := ⟨.program ⟨214⟩, ⟨25472⟩⟩
def mergeEvent : Nat := 8749
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩] } }
def leftRaw : List Term := Proof.Events034.exact8744RawTerms
def rightRaw : List Term := Proof.Events033.exact8558RawTerms
def group : MergeGroup := .operator 8744 8558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8744) (leftOrdinal := 1)
    (rightResult := 8558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8749

namespace LeftMerge8757
def owner : Owner := ⟨.program ⟨214⟩, ⟨29222⟩⟩
def mergeEvent : Nat := 8757
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29220⟩⟩] } }
def leftRaw : List Term := Proof.Events034.exact8751RawTerms
def rightRaw : List Term := Proof.Events033.exact8455RawTerms
def group : MergeGroup := .operator 8751 8455
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8751) (leftOrdinal := 1)
    (rightResult := 8455) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29220⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29220⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8757

namespace LeftMerge8759
def owner : Owner := ⟨.program ⟨214⟩, ⟨29222⟩⟩
def mergeEvent : Nat := 8759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24552⟩⟩] } }
def rhsRaw : List Term := Proof.Events033.exact8452RawTerms
def group : MergeGroup := .relation 8758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 8758) (rhsResult := 8452)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29220⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29220⟩⟩) ⟨24552⟩ 8452) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24552⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16565⟩⟩], [⟨.program ⟨214⟩, ⟨24552⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8759

namespace LeftMerge8760
def owner : Owner := ⟨.program ⟨214⟩, ⟨29222⟩⟩
def mergeEvent : Nat := 8760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29220⟩⟩] } }
def leftRaw : List Term := Proof.Events034.exact8751RawTerms
def rightRaw : List Term := Proof.Events033.exact8455RawTerms
def group : MergeGroup := .operator 8751 8455
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8751) (leftOrdinal := 0)
    (rightResult := 8455) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29220⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29220⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8760

namespace LeftMerge8774
def owner : Owner := ⟨.program ⟨214⟩, ⟨22283⟩⟩
def mergeEvent : Nat := 8774
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22280⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events034.exact8768RawTerms
def group : MergeGroup := .operator 6561 8768
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 8768) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22280⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8774

namespace LeftMerge8895
def owner : Owner := ⟨.program ⟨214⟩, ⟨16607⟩⟩
def mergeEvent : Nat := 8895
def frameStart : Nat := 8829
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events034.exact8891RawTerms
def rightRaw : List Term := Proof.Events034.exact8889RawTerms
def group : MergeGroup := .operator 8891 8889
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8891) (leftOrdinal := 0)
    (rightResult := 8889) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8895

namespace LeftMerge8907
def owner : Owner := ⟨.program ⟨214⟩, ⟨29221⟩⟩
def mergeEvent : Nat := 8907
def frameStart : Nat := 8829
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29220⟩⟩] } }
def leftRaw : List Term := Proof.Events034.exact8903RawTerms
def rightRaw : List Term := Proof.Events034.exact8880RawTerms
def group : MergeGroup := .operator 8903 8880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8903) (leftOrdinal := 1)
    (rightResult := 8880) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29220⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29220⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8907

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
