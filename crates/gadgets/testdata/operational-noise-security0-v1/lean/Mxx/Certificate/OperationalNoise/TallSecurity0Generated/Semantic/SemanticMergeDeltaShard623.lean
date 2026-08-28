import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge101033
def owner : Owner := ⟨.program ⟨214⟩, ⟨10954⟩⟩
def mergeEvent : Nat := 101033
def frameStart : Nat := 101015
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events394.exact101029RawTerms
def rightRaw : List Term := Proof.Events394.exact101026RawTerms
def group : MergeGroup := .operator 101029 101026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101029) (leftOrdinal := 0)
    (rightResult := 101026) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10827⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101033

namespace LeftMerge101063
def owner : Owner := ⟨.program ⟨214⟩, ⟨11067⟩⟩
def mergeEvent : Nat := 101063
def frameStart : Nat := 101015
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events394.exact101059RawTerms
def rightRaw : List Term := Proof.Events394.exact101057RawTerms
def group : MergeGroup := .operator 101059 101057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101059) (leftOrdinal := 0)
    (rightResult := 101057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101063

namespace LeftMerge101086
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def mergeEvent : Nat := 101086
def frameStart : Nat := 101015
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }
def leftRaw : List Term := Proof.Events394.exact101082RawTerms
def rightRaw : List Term := Proof.Events394.exact101079RawTerms
def group : MergeGroup := .operator 101082 101079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101082) (leftOrdinal := 0)
    (rightResult := 101079) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7837⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101086

namespace LeftMerge101095
def owner : Owner := ⟨.program ⟨214⟩, ⟨25055⟩⟩
def mergeEvent : Nat := 101095
def frameStart : Nat := 101015
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩] } }
def leftRaw : List Term := Proof.Events394.exact101091RawTerms
def rightRaw : List Term := Proof.Events394.exact101048RawTerms
def group : MergeGroup := .operator 101091 101048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101091) (leftOrdinal := 0)
    (rightResult := 101048) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25052⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101095

namespace LeftMerge101096
def owner : Owner := ⟨.program ⟨214⟩, ⟨25055⟩⟩
def mergeEvent : Nat := 101096
def frameStart : Nat := 101015
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩] } }
def leftRaw : List Term := Proof.Events394.exact101091RawTerms
def rightRaw : List Term := Proof.Events394.exact101048RawTerms
def group : MergeGroup := .operator 101091 101048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101091) (leftOrdinal := 1)
    (rightResult := 101048) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25052⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101096

namespace LeftMerge101098
def owner : Owner := ⟨.program ⟨214⟩, ⟨25055⟩⟩
def mergeEvent : Nat := 101098
def frameStart : Nat := 101015
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23032⟩⟩] } }
def rhsRaw : List Term := Proof.Events394.exact101045RawTerms
def group : MergeGroup := .relation 101097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101097) (rhsResult := 101045)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25052⟩⟩) ⟨23032⟩ 101045) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23032⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], [⟨.program ⟨214⟩, ⟨23032⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101098

namespace LeftMerge101106
def owner : Owner := ⟨.program ⟨214⟩, ⟨15106⟩⟩
def mergeEvent : Nat := 101106
def frameStart : Nat := 101015
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events394.exact101059RawTerms
def rightRaw : List Term := Proof.Events394.exact101102RawTerms
def group : MergeGroup := .operator 101059 101102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101059) (leftOrdinal := 0)
    (rightResult := 101102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101106

namespace LeftMerge101123
def owner : Owner := ⟨.program ⟨214⟩, ⟨19160⟩⟩
def mergeEvent : Nat := 101123
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101120RawTerms
def group : MergeGroup := .relation 101122
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101122) (rhsResult := 101120)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19157⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101121 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19157⟩⟩]⟩) (none) 101120) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101123

namespace LeftMerge101124
def owner : Owner := ⟨.program ⟨214⟩, ⟨19160⟩⟩
def mergeEvent : Nat := 101124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101120RawTerms
def group : MergeGroup := .relation 101122
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101122) (rhsResult := 101120)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19157⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101121 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19157⟩⟩]⟩) (none) 101120) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101124

namespace LeftMerge101125
def owner : Owner := ⟨.program ⟨214⟩, ⟨19160⟩⟩
def mergeEvent : Nat := 101125
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23032⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101120RawTerms
def group : MergeGroup := .relation 101122
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101122) (rhsResult := 101120)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19157⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101121 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19157⟩⟩]⟩) (none) 101120) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23032⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], [⟨.program ⟨214⟩, ⟨23032⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101125

namespace LeftMerge101126
def owner : Owner := ⟨.program ⟨214⟩, ⟨19160⟩⟩
def mergeEvent : Nat := 101126
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101120RawTerms
def group : MergeGroup := .relation 101122
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101122) (rhsResult := 101120)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19157⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101121 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19157⟩⟩]⟩) (none) 101120) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101126

namespace LeftMerge101131
def owner : Owner := ⟨.program ⟨214⟩, ⟨25054⟩⟩
def mergeEvent : Nat := 101131
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23032⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101127RawTerms
def rightRaw : List Term := Proof.Events394.exact100965RawTerms
def group : MergeGroup := .operator 101127 100965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101127) (leftOrdinal := 2)
    (rightResult := 100965) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23032⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23032⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10827⟩⟩, ⟨.program ⟨214⟩, ⟨10953⟩⟩], [⟨.program ⟨214⟩, ⟨23032⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101131

namespace LeftMerge101132
def owner : Owner := ⟨.program ⟨214⟩, ⟨25054⟩⟩
def mergeEvent : Nat := 101132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101127RawTerms
def rightRaw : List Term := Proof.Events394.exact100965RawTerms
def group : MergeGroup := .operator 101127 100965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101127) (leftOrdinal := 1)
    (rightResult := 100965) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25052⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101132

namespace LeftMerge101140
def owner : Owner := ⟨.program ⟨214⟩, ⟨26748⟩⟩
def mergeEvent : Nat := 101140
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101134RawTerms
def rightRaw : List Term := Proof.Events394.exact100881RawTerms
def group : MergeGroup := .operator 101134 100881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101134) (leftOrdinal := 0)
    (rightResult := 100881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101140

namespace LeftMerge101141
def owner : Owner := ⟨.program ⟨214⟩, ⟨26748⟩⟩
def mergeEvent : Nat := 101141
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101134RawTerms
def rightRaw : List Term := Proof.Events394.exact100881RawTerms
def group : MergeGroup := .operator 101134 100881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101134) (leftOrdinal := 1)
    (rightResult := 100881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101141

namespace LeftMerge101143
def owner : Owner := ⟨.program ⟨214⟩, ⟨26748⟩⟩
def mergeEvent : Nat := 101143
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23838⟩⟩] } }
def rhsRaw : List Term := Proof.Events394.exact100878RawTerms
def group : MergeGroup := .relation 101142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101142) (rhsResult := 100878)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26746⟩⟩) ⟨23838⟩ 100878) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23838⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23838⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101143

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
