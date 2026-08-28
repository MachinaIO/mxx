import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge60787
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60787
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 12)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60787

namespace LeftMerge60788
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60788
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 11)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60788

namespace LeftMerge60789
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60789
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 10)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60789

namespace LeftMerge60790
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60790
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 9)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60790

namespace LeftMerge60791
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60791
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 8)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60791

namespace LeftMerge60792
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60792
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 7)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60792

namespace LeftMerge60793
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60793
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 6)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60793

namespace LeftMerge60794
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60794
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 5)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60794

namespace LeftMerge60795
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60795
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 4)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60795

namespace LeftMerge60796
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60796
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 3)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60796

namespace LeftMerge60797
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60797
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 2)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60797

namespace LeftMerge60798
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60798
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 1)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60798

namespace LeftMerge60799
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60799
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 0)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60799

namespace LeftMerge60800
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60800
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18173⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 33)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18173⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60800

namespace LeftMerge60802
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60802
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18173⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60616RawTerms
def group : MergeGroup := .relation 60801
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60801) (rhsResult := 60616)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 60616) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18173⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60802

namespace LeftMerge60803
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60803
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17088⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 29)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17088⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17088⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60803

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
