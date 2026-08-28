import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge160611
def owner : Owner := ⟨.program ⟨257⟩, ⟨30890⟩⟩
def mergeEvent : Nat := 160611
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩] } }
def leftRaw : List Term := Proof.Events594.exact152198RawTerms
def rightRaw : List Term := Proof.Events627.exact160604RawTerms
def group : MergeGroup := .operator 152198 160604
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152198) (leftOrdinal := 1)
    (rightResult := 160604) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30888⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge160611

namespace LeftMerge160613
def owner : Owner := ⟨.program ⟨257⟩, ⟨30890⟩⟩
def mergeEvent : Nat := 160613
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30213⟩⟩] } }
def rhsRaw : List Term := Proof.Events627.exact160601RawTerms
def group : MergeGroup := .relation 160612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 160612) (rhsResult := 160601)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30888⟩⟩) ⟨30213⟩ 160601) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30213⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨30213⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge160613

namespace LeftMerge160627
def owner : Owner := ⟨.program ⟨257⟩, ⟨29775⟩⟩
def mergeEvent : Nat := 160627
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events627.exact160621RawTerms
def group : MergeGroup := .operator 149120 160621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 160621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29772⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge160627

namespace LeftMerge160748
def owner : Owner := ⟨.program ⟨257⟩, ⟨30436⟩⟩
def mergeEvent : Nat := 160748
def frameStart : Nat := 160682
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events627.exact160744RawTerms
def rightRaw : List Term := Proof.Events627.exact160742RawTerms
def group : MergeGroup := .operator 160744 160742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 160744) (leftOrdinal := 0)
    (rightResult := 160742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge160748

namespace LeftMerge160760
def owner : Owner := ⟨.program ⟨257⟩, ⟨30889⟩⟩
def mergeEvent : Nat := 160760
def frameStart : Nat := 160682
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩] } }
def leftRaw : List Term := Proof.Events627.exact160756RawTerms
def rightRaw : List Term := Proof.Events627.exact160733RawTerms
def group : MergeGroup := .operator 160756 160733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 160756) (leftOrdinal := 0)
    (rightResult := 160733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30888⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge160760

namespace LeftMerge160761
def owner : Owner := ⟨.program ⟨257⟩, ⟨30889⟩⟩
def mergeEvent : Nat := 160761
def frameStart : Nat := 160682
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩] } }
def leftRaw : List Term := Proof.Events627.exact160756RawTerms
def rightRaw : List Term := Proof.Events627.exact160733RawTerms
def group : MergeGroup := .operator 160756 160733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 160756) (leftOrdinal := 1)
    (rightResult := 160733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30888⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge160761

namespace LeftMerge160763
def owner : Owner := ⟨.program ⟨257⟩, ⟨30889⟩⟩
def mergeEvent : Nat := 160763
def frameStart : Nat := 160682
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30213⟩⟩] } }
def rhsRaw : List Term := Proof.Events627.exact160730RawTerms
def group : MergeGroup := .relation 160762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 160762) (rhsResult := 160730)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30888⟩⟩) ⟨30213⟩ 160730) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30213⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨30213⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge160763

namespace LeftMerge160771
def owner : Owner := ⟨.program ⟨257⟩, ⟨29265⟩⟩
def mergeEvent : Nat := 160771
def frameStart : Nat := 160682
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events627.exact160744RawTerms
def rightRaw : List Term := Proof.Events627.exact160767RawTerms
def group : MergeGroup := .operator 160744 160767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 160744) (leftOrdinal := 0)
    (rightResult := 160767) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge160771

namespace LeftMerge160788
def owner : Owner := ⟨.program ⟨257⟩, ⟨29775⟩⟩
def mergeEvent : Nat := 160788
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }
def rhsRaw : List Term := Proof.Events628.exact160785RawTerms
def group : MergeGroup := .relation 160787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 160787) (rhsResult := 160785)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 160786 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩]⟩) (none) 160785) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge160788

namespace LeftMerge160789
def owner : Owner := ⟨.program ⟨257⟩, ⟨29775⟩⟩
def mergeEvent : Nat := 160789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩] } }
def rhsRaw : List Term := Proof.Events628.exact160785RawTerms
def group : MergeGroup := .relation 160787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 160787) (rhsResult := 160785)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 160786 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩]⟩) (none) 160785) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge160789

namespace LeftMerge160790
def owner : Owner := ⟨.program ⟨257⟩, ⟨29775⟩⟩
def mergeEvent : Nat := 160790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30213⟩⟩] } }
def rhsRaw : List Term := Proof.Events628.exact160785RawTerms
def group : MergeGroup := .relation 160787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 160787) (rhsResult := 160785)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 160786 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩]⟩) (none) 160785) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30213⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨30213⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge160790

namespace LeftMerge160791
def owner : Owner := ⟨.program ⟨257⟩, ⟨29775⟩⟩
def mergeEvent : Nat := 160791
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events628.exact160785RawTerms
def group : MergeGroup := .relation 160787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 160787) (rhsResult := 160785)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 160786 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29772⟩⟩]⟩) (none) 160785) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge160791

namespace LeftMerge160796
def owner : Owner := ⟨.program ⟨257⟩, ⟨30891⟩⟩
def mergeEvent : Nat := 160796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩] } }
def leftRaw : List Term := Proof.Events628.exact160792RawTerms
def rightRaw : List Term := Proof.Events627.exact160614RawTerms
def group : MergeGroup := .operator 160792 160614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 160792) (leftOrdinal := 0)
    (rightResult := 160614) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30888⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge160796

namespace LeftMerge160797
def owner : Owner := ⟨.program ⟨257⟩, ⟨30891⟩⟩
def mergeEvent : Nat := 160797
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30213⟩⟩] } }
def leftRaw : List Term := Proof.Events628.exact160792RawTerms
def rightRaw : List Term := Proof.Events627.exact160614RawTerms
def group : MergeGroup := .operator 160792 160614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 160792) (leftOrdinal := 2)
    (rightResult := 160614) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30213⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30213⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨30213⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge160797

namespace LeftMerge160805
def owner : Owner := ⟨.program ⟨257⟩, ⟨30892⟩⟩
def mergeEvent : Nat := 160805
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩] } }
def leftRaw : List Term := Proof.Events628.exact160799RawTerms
def rightRaw : List Term := Proof.Events061.exact15662RawTerms
def group : MergeGroup := .operator 160799 15662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 160799) (leftOrdinal := 0)
    (rightResult := 15662) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7167⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge160805

namespace LeftMerge160806
def owner : Owner := ⟨.program ⟨257⟩, ⟨30892⟩⟩
def mergeEvent : Nat := 160806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩] } }
def leftRaw : List Term := Proof.Events628.exact160799RawTerms
def rightRaw : List Term := Proof.Events061.exact15662RawTerms
def group : MergeGroup := .operator 160799 15662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 160799) (leftOrdinal := 1)
    (rightResult := 15662) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7167⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge160806

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
