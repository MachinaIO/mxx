import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge35001
def owner : Owner := ⟨.program ⟨257⟩, ⟨30699⟩⟩
def mergeEvent : Nat := 35001
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩] } }
def leftRaw : List Term := Proof.Events136.exact34995RawTerms
def rightRaw : List Term := Proof.Events136.exact34931RawTerms
def group : MergeGroup := .operator 34995 34931
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34995) (leftOrdinal := 1)
    (rightResult := 34931) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30698⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35001

namespace LeftMerge35003
def owner : Owner := ⟨.program ⟨257⟩, ⟨30699⟩⟩
def mergeEvent : Nat := 35003
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30143⟩⟩] } }
def rhsRaw : List Term := Proof.Events136.exact34928RawTerms
def group : MergeGroup := .relation 35002
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35002) (rhsResult := 34928)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30698⟩⟩) ⟨30143⟩ 34928) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30143⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], [⟨.program ⟨257⟩, ⟨30143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35003

namespace LeftMerge35004
def owner : Owner := ⟨.program ⟨257⟩, ⟨30699⟩⟩
def mergeEvent : Nat := 35004
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩] } }
def leftRaw : List Term := Proof.Events136.exact34995RawTerms
def rightRaw : List Term := Proof.Events136.exact34931RawTerms
def group : MergeGroup := .operator 34995 34931
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34995) (leftOrdinal := 0)
    (rightResult := 34931) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30698⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35004

namespace LeftMerge35018
def owner : Owner := ⟨.program ⟨257⟩, ⟨29622⟩⟩
def mergeEvent : Nat := 35018
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events136.exact35012RawTerms
def group : MergeGroup := .operator 32120 35012
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 35012) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29619⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35018

namespace LeftMerge35097
def owner : Owner := ⟨.program ⟨257⟩, ⟨28991⟩⟩
def mergeEvent : Nat := 35097
def frameStart : Nat := 35067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events137.exact35093RawTerms
def rightRaw : List Term := Proof.Events137.exact35090RawTerms
def group : MergeGroup := .operator 35093 35090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35093) (leftOrdinal := 0)
    (rightResult := 35090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13416⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35097

namespace LeftMerge35127
def owner : Owner := ⟨.program ⟨257⟩, ⟨30404⟩⟩
def mergeEvent : Nat := 35127
def frameStart : Nat := 35067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events137.exact35123RawTerms
def rightRaw : List Term := Proof.Events137.exact35121RawTerms
def group : MergeGroup := .operator 35123 35121
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35123) (leftOrdinal := 0)
    (rightResult := 35121) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35127

namespace LeftMerge35150
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def mergeEvent : Nat := 35150
def frameStart : Nat := 35067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events137.exact35146RawTerms
def rightRaw : List Term := Proof.Events137.exact35143RawTerms
def group : MergeGroup := .operator 35146 35143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35146) (leftOrdinal := 0)
    (rightResult := 35143) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35150

namespace LeftMerge35159
def owner : Owner := ⟨.program ⟨257⟩, ⟨30701⟩⟩
def mergeEvent : Nat := 35159
def frameStart : Nat := 35067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩] } }
def leftRaw : List Term := Proof.Events137.exact35155RawTerms
def rightRaw : List Term := Proof.Events137.exact35112RawTerms
def group : MergeGroup := .operator 35155 35112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35155) (leftOrdinal := 0)
    (rightResult := 35112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30698⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35159

namespace LeftMerge35160
def owner : Owner := ⟨.program ⟨257⟩, ⟨30701⟩⟩
def mergeEvent : Nat := 35160
def frameStart : Nat := 35067
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩] } }
def leftRaw : List Term := Proof.Events137.exact35155RawTerms
def rightRaw : List Term := Proof.Events137.exact35112RawTerms
def group : MergeGroup := .operator 35155 35112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35155) (leftOrdinal := 1)
    (rightResult := 35112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30698⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35160

namespace LeftMerge35162
def owner : Owner := ⟨.program ⟨257⟩, ⟨30701⟩⟩
def mergeEvent : Nat := 35162
def frameStart : Nat := 35067
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30143⟩⟩] } }
def rhsRaw : List Term := Proof.Events137.exact35109RawTerms
def group : MergeGroup := .relation 35161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35161) (rhsResult := 35109)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30698⟩⟩) ⟨30143⟩ 35109) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30143⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], [⟨.program ⟨257⟩, ⟨30143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35162

namespace LeftMerge35170
def owner : Owner := ⟨.program ⟨257⟩, ⟨29162⟩⟩
def mergeEvent : Nat := 35170
def frameStart : Nat := 35067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events137.exact35123RawTerms
def rightRaw : List Term := Proof.Events137.exact35166RawTerms
def group : MergeGroup := .operator 35123 35166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35123) (leftOrdinal := 0)
    (rightResult := 35166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35170

namespace LeftMerge35187
def owner : Owner := ⟨.program ⟨257⟩, ⟨29622⟩⟩
def mergeEvent : Nat := 35187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events137.exact35184RawTerms
def group : MergeGroup := .relation 35186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35186) (rhsResult := 35184)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩]⟩) (none) 35184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35187

namespace LeftMerge35188
def owner : Owner := ⟨.program ⟨257⟩, ⟨29622⟩⟩
def mergeEvent : Nat := 35188
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩] } }
def rhsRaw : List Term := Proof.Events137.exact35184RawTerms
def group : MergeGroup := .relation 35186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35186) (rhsResult := 35184)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩]⟩) (none) 35184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30698⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35188

namespace LeftMerge35189
def owner : Owner := ⟨.program ⟨257⟩, ⟨29622⟩⟩
def mergeEvent : Nat := 35189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30143⟩⟩] } }
def rhsRaw : List Term := Proof.Events137.exact35184RawTerms
def group : MergeGroup := .relation 35186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35186) (rhsResult := 35184)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩]⟩) (none) 35184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30143⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], [⟨.program ⟨257⟩, ⟨30143⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35189

namespace LeftMerge35190
def owner : Owner := ⟨.program ⟨257⟩, ⟨29622⟩⟩
def mergeEvent : Nat := 35190
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events137.exact35184RawTerms
def group : MergeGroup := .relation 35186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35186) (rhsResult := 35184)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29619⟩⟩]⟩) (none) 35184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35190

namespace LeftMerge35195
def owner : Owner := ⟨.program ⟨257⟩, ⟨30700⟩⟩
def mergeEvent : Nat := 35195
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30143⟩⟩] } }
def leftRaw : List Term := Proof.Events137.exact35191RawTerms
def rightRaw : List Term := Proof.Events136.exact35005RawTerms
def group : MergeGroup := .operator 35191 35005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35191) (leftOrdinal := 2)
    (rightResult := 35005) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30143⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30143⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13416⟩⟩, ⟨.program ⟨257⟩, ⟨28990⟩⟩], [⟨.program ⟨257⟩, ⟨30143⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35195

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
