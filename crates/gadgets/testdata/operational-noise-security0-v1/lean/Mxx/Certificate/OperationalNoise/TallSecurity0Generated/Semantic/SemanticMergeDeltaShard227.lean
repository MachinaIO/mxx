import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge37730
def owner : Owner := ⟨.program ⟨214⟩, ⟨25540⟩⟩
def mergeEvent : Nat := 37730
def frameStart : Nat := 37638
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩] } }
def leftRaw : List Term := Proof.Events147.exact37726RawTerms
def rightRaw : List Term := Proof.Events147.exact37683RawTerms
def group : MergeGroup := .operator 37726 37683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37726) (leftOrdinal := 0)
    (rightResult := 37683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25537⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37730

namespace LeftMerge37731
def owner : Owner := ⟨.program ⟨214⟩, ⟨25540⟩⟩
def mergeEvent : Nat := 37731
def frameStart : Nat := 37638
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩] } }
def leftRaw : List Term := Proof.Events147.exact37726RawTerms
def rightRaw : List Term := Proof.Events147.exact37683RawTerms
def group : MergeGroup := .operator 37726 37683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37726) (leftOrdinal := 1)
    (rightResult := 37683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25537⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37731

namespace LeftMerge37733
def owner : Owner := ⟨.program ⟨214⟩, ⟨25540⟩⟩
def mergeEvent : Nat := 37733
def frameStart : Nat := 37638
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23294⟩⟩] } }
def rhsRaw : List Term := Proof.Events147.exact37680RawTerms
def group : MergeGroup := .relation 37732
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37732) (rhsResult := 37680)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25537⟩⟩) ⟨23294⟩ 37680) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23294⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], [⟨.program ⟨214⟩, ⟨23294⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37733

namespace LeftMerge37741
def owner : Owner := ⟨.program ⟨214⟩, ⟨16643⟩⟩
def mergeEvent : Nat := 37741
def frameStart : Nat := 37638
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events147.exact37694RawTerms
def rightRaw : List Term := Proof.Events147.exact37737RawTerms
def group : MergeGroup := .operator 37694 37737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37694) (leftOrdinal := 0)
    (rightResult := 37737) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16641⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37741

namespace LeftMerge37758
def owner : Owner := ⟨.program ⟨214⟩, ⟨20043⟩⟩
def mergeEvent : Nat := 37758
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }
def rhsRaw : List Term := Proof.Events147.exact37755RawTerms
def group : MergeGroup := .relation 37757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37757) (rhsResult := 37755)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20040⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37756 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20040⟩⟩]⟩) (none) 37755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37758

namespace LeftMerge37759
def owner : Owner := ⟨.program ⟨214⟩, ⟨20043⟩⟩
def mergeEvent : Nat := 37759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩] } }
def rhsRaw : List Term := Proof.Events147.exact37755RawTerms
def group : MergeGroup := .relation 37757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37757) (rhsResult := 37755)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20040⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37756 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20040⟩⟩]⟩) (none) 37755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37759

namespace LeftMerge37760
def owner : Owner := ⟨.program ⟨214⟩, ⟨20043⟩⟩
def mergeEvent : Nat := 37760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23294⟩⟩] } }
def rhsRaw : List Term := Proof.Events147.exact37755RawTerms
def group : MergeGroup := .relation 37757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37757) (rhsResult := 37755)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20040⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37756 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20040⟩⟩]⟩) (none) 37755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23294⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], [⟨.program ⟨214⟩, ⟨23294⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37760

namespace LeftMerge37761
def owner : Owner := ⟨.program ⟨214⟩, ⟨20043⟩⟩
def mergeEvent : Nat := 37761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events147.exact37755RawTerms
def group : MergeGroup := .relation 37757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37757) (rhsResult := 37755)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20040⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37756 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20040⟩⟩]⟩) (none) 37755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37761

namespace LeftMerge37766
def owner : Owner := ⟨.program ⟨214⟩, ⟨25539⟩⟩
def mergeEvent : Nat := 37766
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23294⟩⟩] } }
def leftRaw : List Term := Proof.Events147.exact37762RawTerms
def rightRaw : List Term := Proof.Events146.exact37576RawTerms
def group : MergeGroup := .operator 37762 37576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37762) (leftOrdinal := 2)
    (rightResult := 37576) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23294⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23294⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], [⟨.program ⟨214⟩, ⟨23294⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37766

namespace LeftMerge37767
def owner : Owner := ⟨.program ⟨214⟩, ⟨25539⟩⟩
def mergeEvent : Nat := 37767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩] } }
def leftRaw : List Term := Proof.Events147.exact37762RawTerms
def rightRaw : List Term := Proof.Events146.exact37576RawTerms
def group : MergeGroup := .operator 37762 37576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37762) (leftOrdinal := 1)
    (rightResult := 37576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37767

namespace LeftMerge37775
def owner : Owner := ⟨.program ⟨214⟩, ⟨29413⟩⟩
def mergeEvent : Nat := 37775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29411⟩⟩] } }
def leftRaw : List Term := Proof.Events147.exact37769RawTerms
def rightRaw : List Term := Proof.Events146.exact37492RawTerms
def group : MergeGroup := .operator 37769 37492
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37769) (leftOrdinal := 0)
    (rightResult := 37492) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29411⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29411⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37775

namespace LeftMerge37776
def owner : Owner := ⟨.program ⟨214⟩, ⟨29413⟩⟩
def mergeEvent : Nat := 37776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29411⟩⟩] } }
def leftRaw : List Term := Proof.Events147.exact37769RawTerms
def rightRaw : List Term := Proof.Events146.exact37492RawTerms
def group : MergeGroup := .operator 37769 37492
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37769) (leftOrdinal := 1)
    (rightResult := 37492) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29411⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29411⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37776

namespace LeftMerge37778
def owner : Owner := ⟨.program ⟨214⟩, ⟨29413⟩⟩
def mergeEvent : Nat := 37778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24609⟩⟩] } }
def rhsRaw : List Term := Proof.Events146.exact37489RawTerms
def group : MergeGroup := .relation 37777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37777) (rhsResult := 37489)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29411⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29411⟩⟩) ⟨24609⟩ 37489) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24609⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16641⟩⟩], [⟨.program ⟨214⟩, ⟨24609⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37778

namespace LeftMerge37792
def owner : Owner := ⟨.program ⟨214⟩, ⟨22419⟩⟩
def mergeEvent : Nat := 37792
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22416⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events147.exact37786RawTerms
def group : MergeGroup := .operator 36137 37786
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 37786) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22416⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22416⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37792

namespace LeftMerge37913
def owner : Owner := ⟨.program ⟨214⟩, ⟨16718⟩⟩
def mergeEvent : Nat := 37913
def frameStart : Nat := 37847
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact37909RawTerms
def rightRaw : List Term := Proof.Events148.exact37907RawTerms
def group : MergeGroup := .operator 37909 37907
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37909) (leftOrdinal := 0)
    (rightResult := 37907) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16641⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37913

namespace LeftMerge37925
def owner : Owner := ⟨.program ⟨214⟩, ⟨29412⟩⟩
def mergeEvent : Nat := 37925
def frameStart : Nat := 37847
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29411⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact37921RawTerms
def rightRaw : List Term := Proof.Events148.exact37898RawTerms
def group : MergeGroup := .operator 37921 37898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37921) (leftOrdinal := 0)
    (rightResult := 37898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29411⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29411⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37925

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
