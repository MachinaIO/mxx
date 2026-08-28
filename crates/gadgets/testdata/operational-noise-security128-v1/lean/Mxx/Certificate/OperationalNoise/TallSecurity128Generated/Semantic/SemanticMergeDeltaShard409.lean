import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge70068
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70068
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 9)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70068

namespace LeftMerge70069
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70069
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 35)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70069

namespace LeftMerge70071
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70071
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70070
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70070) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70071

namespace LeftMerge70072
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70072
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 8)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70072

namespace LeftMerge70073
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70073
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 34)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70073

namespace LeftMerge70075
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70075
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70074
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70074) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70075

namespace LeftMerge70076
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70076
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 7)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70076

namespace LeftMerge70077
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70077
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 33)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70077

namespace LeftMerge70079
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70078) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70079

namespace LeftMerge70080
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 6)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70080

namespace LeftMerge70081
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 32)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70081

namespace LeftMerge70083
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70083
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70082) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70083

namespace LeftMerge70084
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70084
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 5)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70084

namespace LeftMerge70085
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70085
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 31)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70085

namespace LeftMerge70087
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def rhsRaw : List Term := Proof.Events239.exact61250RawTerms
def group : MergeGroup := .relation 70086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 70086) (rhsResult := 61250)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge70087

namespace LeftMerge70088
def owner : Owner := ⟨.program ⟨257⟩, ⟨71471⟩⟩
def mergeEvent : Nat := 70088
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact70030RawTerms
def rightRaw : List Term := Proof.Events239.exact61253RawTerms
def group : MergeGroup := .operator 70030 61253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 70030) (leftOrdinal := 4)
    (rightResult := 61253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge70088

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
