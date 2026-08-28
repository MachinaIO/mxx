import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge32084
def owner : Owner := ⟨.program ⟨257⟩, ⟨15220⟩⟩
def mergeEvent : Nat := 32084
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def rhsRaw : List Term := Proof.Events066.exact17065RawTerms
def group : MergeGroup := .relation 32083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 32083) (rhsResult := 17065)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9565⟩⟩) ⟨7285⟩ 17065) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32084

namespace LeftMerge32085
def owner : Owner := ⟨.program ⟨257⟩, ⟨15220⟩⟩
def mergeEvent : Nat := 32085
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32076RawTerms
def rightRaw : List Term := Proof.Events066.exact17095RawTerms
def group : MergeGroup := .operator 32076 17095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32076) (leftOrdinal := 0)
    (rightResult := 17095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32085

namespace LeftMerge32090
def owner : Owner := ⟨.program ⟨257⟩, ⟨48057⟩⟩
def mergeEvent : Nat := 32090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32086RawTerms
def rightRaw : List Term := Proof.Events125.exact32056RawTerms
def group : MergeGroup := .operator 32086 32056
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32086) (leftOrdinal := 1)
    (rightResult := 32056) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32090

namespace LeftMerge32098
def owner : Owner := ⟨.program ⟨257⟩, ⟨49759⟩⟩
def mergeEvent : Nat := 32098
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32092RawTerms
def rightRaw : List Term := Proof.Events125.exact32023RawTerms
def group : MergeGroup := .operator 32092 32023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32092) (leftOrdinal := 1)
    (rightResult := 32023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49758⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32098

namespace LeftMerge32100
def owner : Owner := ⟨.program ⟨257⟩, ⟨49759⟩⟩
def mergeEvent : Nat := 32100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49203⟩⟩] } }
def rhsRaw : List Term := Proof.Events125.exact32020RawTerms
def group : MergeGroup := .relation 32099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 32099) (rhsResult := 32020)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49758⟩⟩) ⟨49203⟩ 32020) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49203⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], [⟨.program ⟨257⟩, ⟨49203⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32100

namespace LeftMerge32101
def owner : Owner := ⟨.program ⟨257⟩, ⟨49759⟩⟩
def mergeEvent : Nat := 32101
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32092RawTerms
def rightRaw : List Term := Proof.Events125.exact32023RawTerms
def group : MergeGroup := .operator 32092 32023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32092) (leftOrdinal := 0)
    (rightResult := 32023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49758⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32101

namespace LeftMerge32113
def owner : Owner := ⟨.program ⟨257⟩, ⟨11642⟩⟩
def mergeEvent : Nat := 32113
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }
def leftRaw : List Term := Proof.Events124.exact31898RawTerms
def rightRaw : List Term := Proof.Events067.exact17158RawTerms
def group : MergeGroup := .operator 31898 17158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31898) (leftOrdinal := 0)
    (rightResult := 17158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32113

namespace LeftMerge32126
def owner : Owner := ⟨.program ⟨257⟩, ⟨48682⟩⟩
def mergeEvent : Nat := 32126
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48679⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events125.exact32109RawTerms
def group : MergeGroup := .operator 32120 32109
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 32109) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48679⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48679⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32126

namespace LeftMerge32205
def owner : Owner := ⟨.program ⟨257⟩, ⟨48051⟩⟩
def mergeEvent : Nat := 32205
def frameStart : Nat := 32175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events125.exact32201RawTerms
def rightRaw : List Term := Proof.Events125.exact32198RawTerms
def group : MergeGroup := .operator 32201 32198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32201) (leftOrdinal := 0)
    (rightResult := 32198) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15216⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32205

namespace LeftMerge32235
def owner : Owner := ⟨.program ⟨257⟩, ⟨49464⟩⟩
def mergeEvent : Nat := 32235
def frameStart : Nat := 32175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32231RawTerms
def rightRaw : List Term := Proof.Events125.exact32229RawTerms
def group : MergeGroup := .operator 32231 32229
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32231) (leftOrdinal := 0)
    (rightResult := 32229) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32235

namespace LeftMerge32258
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 32258
def frameStart : Nat := 32175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32254RawTerms
def rightRaw : List Term := Proof.Events125.exact32251RawTerms
def group : MergeGroup := .operator 32254 32251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32254) (leftOrdinal := 0)
    (rightResult := 32251) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32258

namespace LeftMerge32267
def owner : Owner := ⟨.program ⟨257⟩, ⟨49761⟩⟩
def mergeEvent : Nat := 32267
def frameStart : Nat := 32175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩] } }
def leftRaw : List Term := Proof.Events126.exact32263RawTerms
def rightRaw : List Term := Proof.Events125.exact32220RawTerms
def group : MergeGroup := .operator 32263 32220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32263) (leftOrdinal := 0)
    (rightResult := 32220) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49758⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32267

namespace LeftMerge32268
def owner : Owner := ⟨.program ⟨257⟩, ⟨49761⟩⟩
def mergeEvent : Nat := 32268
def frameStart : Nat := 32175
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩] } }
def leftRaw : List Term := Proof.Events126.exact32263RawTerms
def rightRaw : List Term := Proof.Events125.exact32220RawTerms
def group : MergeGroup := .operator 32263 32220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32263) (leftOrdinal := 1)
    (rightResult := 32220) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49758⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32268

namespace LeftMerge32270
def owner : Owner := ⟨.program ⟨257⟩, ⟨49761⟩⟩
def mergeEvent : Nat := 32270
def frameStart : Nat := 32175
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49203⟩⟩] } }
def rhsRaw : List Term := Proof.Events125.exact32217RawTerms
def group : MergeGroup := .relation 32269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 32269) (rhsResult := 32217)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49758⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49758⟩⟩) ⟨49203⟩ 32217) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49203⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15216⟩⟩, ⟨.program ⟨257⟩, ⟨48050⟩⟩], [⟨.program ⟨257⟩, ⟨49203⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32270

namespace LeftMerge32278
def owner : Owner := ⟨.program ⟨257⟩, ⟨48222⟩⟩
def mergeEvent : Nat := 32278
def frameStart : Nat := 32175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48220⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32231RawTerms
def rightRaw : List Term := Proof.Events126.exact32274RawTerms
def group : MergeGroup := .operator 32231 32274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32231) (leftOrdinal := 0)
    (rightResult := 32274) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48220⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32278

namespace LeftMerge32295
def owner : Owner := ⟨.program ⟨257⟩, ⟨48682⟩⟩
def mergeEvent : Nat := 32295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events126.exact32292RawTerms
def group : MergeGroup := .relation 32294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 32294) (rhsResult := 32292)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48679⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 32293 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48679⟩⟩]⟩) (none) 32292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32295

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
