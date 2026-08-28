import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge22057
def owner : Owner := ⟨.program ⟨257⟩, ⟨64605⟩⟩
def mergeEvent : Nat := 22057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22052RawTerms
def rightRaw : List Term := Proof.Events085.exact21874RawTerms
def group : MergeGroup := .operator 22052 21874
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22052) (leftOrdinal := 0)
    (rightResult := 21874) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22057

namespace LeftMerge22086
def owner : Owner := ⟨.program ⟨257⟩, ⟨25147⟩⟩
def mergeEvent : Nat := 22086
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events001.exact281RawTerms
def rightRaw : List Term := Proof.Events066.exact17057RawTerms
def group : MergeGroup := .operator 281 17057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 281) (leftOrdinal := 0)
    (rightResult := 17057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25146⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22086

namespace LeftMerge22094
def owner : Owner := ⟨.program ⟨257⟩, ⟨7592⟩⟩
def mergeEvent : Nat := 22094
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16922RawTerms
def rightRaw : List Term := Proof.Events086.exact22090RawTerms
def group : MergeGroup := .operator 16922 22090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16922) (leftOrdinal := 0)
    (rightResult := 22090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22094

namespace LeftMerge22111
def owner : Owner := ⟨.program ⟨257⟩, ⟨59254⟩⟩
def mergeEvent : Nat := 22111
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22105RawTerms
def rightRaw : List Term := Proof.Events001.exact284RawTerms
def group : MergeGroup := .operator 22105 284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22105) (leftOrdinal := 1)
    (rightResult := 284) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22111

namespace LeftMerge22112
def owner : Owner := ⟨.program ⟨257⟩, ⟨59254⟩⟩
def mergeEvent : Nat := 22112
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22105RawTerms
def rightRaw : List Term := Proof.Events001.exact284RawTerms
def group : MergeGroup := .operator 22105 284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22105) (leftOrdinal := 0)
    (rightResult := 284) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22112

namespace LeftMerge22127
def owner : Owner := ⟨.program ⟨257⟩, ⟨59255⟩⟩
def mergeEvent : Nat := 22127
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events001.exact284RawTerms
def rightRaw : List Term := Proof.Events066.exact17057RawTerms
def group : MergeGroup := .operator 284 17057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284) (leftOrdinal := 0)
    (rightResult := 17057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22127

namespace LeftMerge22135
def owner : Owner := ⟨.program ⟨257⟩, ⟨7609⟩⟩
def mergeEvent : Nat := 22135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16922RawTerms
def rightRaw : List Term := Proof.Events086.exact22131RawTerms
def group : MergeGroup := .operator 16922 22131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16922) (leftOrdinal := 0)
    (rightResult := 22131) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22135

namespace LeftMerge22152
def owner : Owner := ⟨.program ⟨257⟩, ⟨59258⟩⟩
def mergeEvent : Nat := 22152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22146RawTerms
def rightRaw : List Term := Proof.Events086.exact22120RawTerms
def group : MergeGroup := .operator 22146 22120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22146) (leftOrdinal := 1)
    (rightResult := 22120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22152

namespace LeftMerge22154
def owner : Owner := ⟨.program ⟨257⟩, ⟨59258⟩⟩
def mergeEvent : Nat := 22154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def rhsRaw : List Term := Proof.Events086.exact22090RawTerms
def group : MergeGroup := .relation 22153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22153) (rhsResult := 22090)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9535⟩⟩) ⟨7274⟩ 22090) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22154

namespace LeftMerge22155
def owner : Owner := ⟨.program ⟨257⟩, ⟨59258⟩⟩
def mergeEvent : Nat := 22155
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22146RawTerms
def rightRaw : List Term := Proof.Events086.exact22120RawTerms
def group : MergeGroup := .operator 22146 22120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22146) (leftOrdinal := 0)
    (rightResult := 22120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22155

namespace LeftMerge22160
def owner : Owner := ⟨.program ⟨257⟩, ⟨59259⟩⟩
def mergeEvent : Nat := 22160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22156RawTerms
def rightRaw : List Term := Proof.Events086.exact22113RawTerms
def group : MergeGroup := .operator 22156 22113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22156) (leftOrdinal := 1)
    (rightResult := 22113) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22160

namespace LeftMerge22168
def owner : Owner := ⟨.program ⟨257⟩, ⟨61364⟩⟩
def mergeEvent : Nat := 22168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61363⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22162RawTerms
def rightRaw : List Term := Proof.Events086.exact22079RawTerms
def group : MergeGroup := .operator 22162 22079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22162) (leftOrdinal := 1)
    (rightResult := 22079) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61363⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61363⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22168

namespace LeftMerge22170
def owner : Owner := ⟨.program ⟨257⟩, ⟨61364⟩⟩
def mergeEvent : Nat := 22170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60897⟩⟩] } }
def rhsRaw : List Term := Proof.Events086.exact22076RawTerms
def group : MergeGroup := .relation 22169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22169) (rhsResult := 22076)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61363⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61363⟩⟩) ⟨60897⟩ 22076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60897⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], [⟨.program ⟨257⟩, ⟨60897⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22170

namespace LeftMerge22171
def owner : Owner := ⟨.program ⟨257⟩, ⟨61364⟩⟩
def mergeEvent : Nat := 22171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61363⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22162RawTerms
def rightRaw : List Term := Proof.Events086.exact22079RawTerms
def group : MergeGroup := .operator 22162 22079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22162) (leftOrdinal := 0)
    (rightResult := 22079) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61363⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61363⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22171

namespace LeftMerge22185
def owner : Owner := ⟨.program ⟨257⟩, ⟨60305⟩⟩
def mergeEvent : Nat := 22185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60302⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events086.exact22179RawTerms
def group : MergeGroup := .operator 17169 22179
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 22179) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60302⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60302⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22185

namespace LeftMerge22264
def owner : Owner := ⟨.program ⟨257⟩, ⟨59252⟩⟩
def mergeEvent : Nat := 22264
def frameStart : Nat := 22234
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events086.exact22260RawTerms
def rightRaw : List Term := Proof.Events086.exact22257RawTerms
def group : MergeGroup := .operator 22260 22257
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22260) (leftOrdinal := 0)
    (rightResult := 22257) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59251⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25146⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25146⟩⟩, ⟨.program ⟨257⟩, ⟨59251⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22264

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
