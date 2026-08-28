import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge40789
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨48480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events125.exact32000RawTerms
def group : MergeGroup := .relation 40788
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40788) (rhsResult := 32000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨48480⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40789

namespace LeftMerge40790
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 16)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40790

namespace LeftMerge40791
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40791
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 28)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40791

namespace LeftMerge40793
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40793
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events125.exact32000RawTerms
def group : MergeGroup := .relation 40792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40792) (rhsResult := 32000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45800⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40793

namespace LeftMerge40794
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40794
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 15)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40794

namespace LeftMerge40795
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 27)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40795

namespace LeftMerge40797
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40797
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events125.exact32000RawTerms
def group : MergeGroup := .relation 40796
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40796) (rhsResult := 32000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43116⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40797

namespace LeftMerge40798
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40798
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 14)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40798

namespace LeftMerge40799
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40799
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 26)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40799

namespace LeftMerge40801
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events125.exact32000RawTerms
def group : MergeGroup := .relation 40800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40800) (rhsResult := 32000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40801

namespace LeftMerge40802
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40802
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 13)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40802

namespace LeftMerge40803
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 25)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40803

namespace LeftMerge40805
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40805
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events125.exact32000RawTerms
def group : MergeGroup := .relation 40804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40804) (rhsResult := 32000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37760⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40805

namespace LeftMerge40806
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 12)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40806

namespace LeftMerge40807
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40807
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events159.exact40780RawTerms
def rightRaw : List Term := Proof.Events125.exact32003RawTerms
def group : MergeGroup := .operator 40780 32003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40780) (leftOrdinal := 24)
    (rightResult := 32003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40807

namespace LeftMerge40809
def owner : Owner := ⟨.program ⟨257⟩, ⟨71536⟩⟩
def mergeEvent : Nat := 40809
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }
def rhsRaw : List Term := Proof.Events125.exact32000RawTerms
def group : MergeGroup := .relation 40808
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40808) (rhsResult := 32000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71534⟩⟩) ⟨68884⟩ 32000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68884⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨68884⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40809

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
