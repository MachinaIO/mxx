import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge21939
def owner : Owner := ⟨.program ⟨214⟩, ⟨13184⟩⟩
def mergeEvent : Nat := 21939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21933RawTerms
def rightRaw : List Term := Proof.Events003.exact868RawTerms
def group : MergeGroup := .operator 21933 868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21933) (leftOrdinal := 1)
    (rightResult := 868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21939

namespace LeftMerge21940
def owner : Owner := ⟨.program ⟨214⟩, ⟨13184⟩⟩
def mergeEvent : Nat := 21940
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21933RawTerms
def rightRaw : List Term := Proof.Events003.exact868RawTerms
def group : MergeGroup := .operator 21933 868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21933) (leftOrdinal := 0)
    (rightResult := 868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21940

namespace LeftMerge21945
def owner : Owner := ⟨.program ⟨214⟩, ⟨10256⟩⟩
def mergeEvent : Nat := 21945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events003.exact868RawTerms
def rightRaw : List Term := Proof.Events083.exact21420RawTerms
def group : MergeGroup := .operator 868 21420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 868) (leftOrdinal := 0)
    (rightResult := 21420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21945

namespace LeftMerge21950
def owner : Owner := ⟨.program ⟨214⟩, ⟨7339⟩⟩
def mergeEvent : Nat := 21950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21290RawTerms
def rightRaw : List Term := Proof.Events027.exact7014RawTerms
def group : MergeGroup := .operator 21290 7014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21290) (leftOrdinal := 0)
    (rightResult := 7014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21950

namespace LeftMerge21967
def owner : Owner := ⟨.program ⟨214⟩, ⟨10259⟩⟩
def mergeEvent : Nat := 21967
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21961RawTerms
def rightRaw : List Term := Proof.Events027.exact7003RawTerms
def group : MergeGroup := .operator 21961 7003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21961) (leftOrdinal := 1)
    (rightResult := 7003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21967

namespace LeftMerge21969
def owner : Owner := ⟨.program ⟨214⟩, ⟨10259⟩⟩
def mergeEvent : Nat := 21969
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact6973RawTerms
def group : MergeGroup := .relation 21968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21968) (rhsResult := 6973)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7879⟩⟩) ⟨6789⟩ 6973) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21969

namespace LeftMerge21970
def owner : Owner := ⟨.program ⟨214⟩, ⟨10259⟩⟩
def mergeEvent : Nat := 21970
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21961RawTerms
def rightRaw : List Term := Proof.Events027.exact7003RawTerms
def group : MergeGroup := .operator 21961 7003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21961) (leftOrdinal := 0)
    (rightResult := 7003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21970

namespace LeftMerge21975
def owner : Owner := ⟨.program ⟨214⟩, ⟨13185⟩⟩
def mergeEvent : Nat := 21975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21971RawTerms
def rightRaw : List Term := Proof.Events085.exact21941RawTerms
def group : MergeGroup := .operator 21971 21941
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21971) (leftOrdinal := 1)
    (rightResult := 21941) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21975

namespace LeftMerge21983
def owner : Owner := ⟨.program ⟨214⟩, ⟨25697⟩⟩
def mergeEvent : Nat := 21983
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21977RawTerms
def rightRaw : List Term := Proof.Events085.exact21913RawTerms
def group : MergeGroup := .operator 21977 21913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21977) (leftOrdinal := 1)
    (rightResult := 21913) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25696⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21983

namespace LeftMerge21985
def owner : Owner := ⟨.program ⟨214⟩, ⟨25697⟩⟩
def mergeEvent : Nat := 21985
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23380⟩⟩] } }
def rhsRaw : List Term := Proof.Events085.exact21910RawTerms
def group : MergeGroup := .relation 21984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21984) (rhsResult := 21910)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25696⟩⟩) ⟨23380⟩ 21910) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23380⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], [⟨.program ⟨214⟩, ⟨23380⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21985

namespace LeftMerge21986
def owner : Owner := ⟨.program ⟨214⟩, ⟨25697⟩⟩
def mergeEvent : Nat := 21986
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21977RawTerms
def rightRaw : List Term := Proof.Events085.exact21913RawTerms
def group : MergeGroup := .operator 21977 21913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21977) (leftOrdinal := 0)
    (rightResult := 21913) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25696⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21986

namespace LeftMerge22000
def owner : Owner := ⟨.program ⟨214⟩, ⟨20191⟩⟩
def mergeEvent : Nat := 22000
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20188⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events085.exact21994RawTerms
def group : MergeGroup := .operator 21512 21994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 21994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20188⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22000

namespace LeftMerge22079
def owner : Owner := ⟨.program ⟨214⟩, ⟨13179⟩⟩
def mergeEvent : Nat := 22079
def frameStart : Nat := 22049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events086.exact22075RawTerms
def rightRaw : List Term := Proof.Events086.exact22072RawTerms
def group : MergeGroup := .operator 22075 22072
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22075) (leftOrdinal := 0)
    (rightResult := 22072) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22079

namespace LeftMerge22109
def owner : Owner := ⟨.program ⟨214⟩, ⟨13264⟩⟩
def mergeEvent : Nat := 22109
def frameStart : Nat := 22049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22105RawTerms
def rightRaw : List Term := Proof.Events086.exact22103RawTerms
def group : MergeGroup := .operator 22105 22103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22105) (leftOrdinal := 0)
    (rightResult := 22103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22109

namespace LeftMerge22132
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def mergeEvent : Nat := 22132
def frameStart : Nat := 22049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22128RawTerms
def rightRaw : List Term := Proof.Events086.exact22125RawTerms
def group : MergeGroup := .operator 22128 22125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22128) (leftOrdinal := 0)
    (rightResult := 22125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22132

namespace LeftMerge22141
def owner : Owner := ⟨.program ⟨214⟩, ⟨25699⟩⟩
def mergeEvent : Nat := 22141
def frameStart : Nat := 22049
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22137RawTerms
def rightRaw : List Term := Proof.Events086.exact22094RawTerms
def group : MergeGroup := .operator 22137 22094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22137) (leftOrdinal := 0)
    (rightResult := 22094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25696⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22141

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
