import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge86033
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86033
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48441⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 29)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48441⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86033

namespace LeftMerge86035
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86035
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48441⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85849RawTerms
def group : MergeGroup := .relation 86034
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86034) (rhsResult := 85849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 85849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86035

namespace LeftMerge86036
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86036
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 28)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86036

namespace LeftMerge86038
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86038
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85849RawTerms
def group : MergeGroup := .relation 86037
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86037) (rhsResult := 85849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 85849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86038

namespace LeftMerge86039
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86039
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 27)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86039

namespace LeftMerge86041
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86041
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85849RawTerms
def group : MergeGroup := .relation 86040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86040) (rhsResult := 85849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 85849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86041

namespace LeftMerge86042
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86042
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 26)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86042

namespace LeftMerge86044
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86044
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85849RawTerms
def group : MergeGroup := .relation 86043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86043) (rhsResult := 85849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 85849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86044

namespace LeftMerge86045
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86045
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 25)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86045

namespace LeftMerge86047
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86047
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85849RawTerms
def group : MergeGroup := .relation 86046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86046) (rhsResult := 85849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 85849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86047

namespace LeftMerge86048
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86048
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 24)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86048

namespace LeftMerge86050
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86050
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85849RawTerms
def group : MergeGroup := .relation 86049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86049) (rhsResult := 85849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 85849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86050

namespace LeftMerge86051
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86051
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 22)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86051

namespace LeftMerge86053
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86053
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85849RawTerms
def group : MergeGroup := .relation 86052
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86052) (rhsResult := 85849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 85849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86053

namespace LeftMerge86054
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86054
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 21)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86054

namespace LeftMerge86056
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86056
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85849RawTerms
def group : MergeGroup := .relation 86055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86055) (rhsResult := 85849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 85849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86056

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
