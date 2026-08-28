import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge70047
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70046) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70047

namespace LeftMerge70048
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70048
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 14)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70048

namespace LeftMerge70049
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 26)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70049

namespace LeftMerge70051
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70050
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70050) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70051

namespace LeftMerge70052
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 13)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70052

namespace LeftMerge70053
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 25)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70053

namespace LeftMerge70055
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70054) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70055

namespace LeftMerge70056
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 12)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70056

namespace LeftMerge70057
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 24)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70057

namespace LeftMerge70059
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70058) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70059

namespace LeftMerge70060
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 11)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70060

namespace LeftMerge70061
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 22)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70061

namespace LeftMerge70063
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70062
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70062) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70063

namespace LeftMerge70064
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70064
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 10)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70064

namespace LeftMerge70065
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70065
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 21)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70065

namespace LeftMerge70067
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70067
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70066
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70066) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70067

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
