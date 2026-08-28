import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge75572
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def mergeEvent : Nat := 75572
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75541RawTerms
def rightRaw : List Term := Proof.Events289.exact74125RawTerms
def group : MergeGroup := .operator 75541 74125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75541) (leftOrdinal := 22)
    (rightResult := 74125) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15626⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75572

namespace LeftMerge75573
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def mergeEvent : Nat := 75573
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75541RawTerms
def rightRaw : List Term := Proof.Events289.exact74125RawTerms
def group : MergeGroup := .operator 75541 74125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75541) (leftOrdinal := 3)
    (rightResult := 74125) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75573

namespace LeftMerge75574
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def mergeEvent : Nat := 75574
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75541RawTerms
def rightRaw : List Term := Proof.Events289.exact74125RawTerms
def group : MergeGroup := .operator 75541 74125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75541) (leftOrdinal := 32)
    (rightResult := 74125) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17318⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75574

namespace LeftMerge75575
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def mergeEvent : Nat := 75575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75541RawTerms
def rightRaw : List Term := Proof.Events289.exact74125RawTerms
def group : MergeGroup := .operator 75541 74125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75541) (leftOrdinal := 2)
    (rightResult := 74125) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75575

namespace LeftMerge75576
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def mergeEvent : Nat := 75576
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75541RawTerms
def rightRaw : List Term := Proof.Events289.exact74125RawTerms
def group : MergeGroup := .operator 75541 74125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75541) (leftOrdinal := 21)
    (rightResult := 74125) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75576

namespace LeftMerge75577
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def mergeEvent : Nat := 75577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75541RawTerms
def rightRaw : List Term := Proof.Events289.exact74125RawTerms
def group : MergeGroup := .operator 75541 74125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75541) (leftOrdinal := 1)
    (rightResult := 74125) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75577

namespace LeftMerge75578
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def mergeEvent : Nat := 75578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75541RawTerms
def rightRaw : List Term := Proof.Events289.exact74125RawTerms
def group : MergeGroup := .operator 75541 74125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75541) (leftOrdinal := 20)
    (rightResult := 74125) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75578

namespace LeftMerge75579
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def mergeEvent : Nat := 75579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75541RawTerms
def rightRaw : List Term := Proof.Events289.exact74125RawTerms
def group : MergeGroup := .operator 75541 74125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75541) (leftOrdinal := 0)
    (rightResult := 74125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75579

namespace LeftMerge75580
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def mergeEvent : Nat := 75580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75541RawTerms
def rightRaw : List Term := Proof.Events289.exact74125RawTerms
def group : MergeGroup := .operator 75541 74125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75541) (leftOrdinal := 19)
    (rightResult := 74125) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75580

namespace LeftMerge75588
def owner : Owner := ⟨.program ⟨214⟩, ⟨30102⟩⟩
def mergeEvent : Nat := 75588
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75582RawTerms
def rightRaw : List Term := Proof.Events021.exact5499RawTerms
def group : MergeGroup := .operator 75582 5499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75582) (leftOrdinal := 0)
    (rightResult := 5499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6651⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75588

namespace LeftMerge75589
def owner : Owner := ⟨.program ⟨214⟩, ⟨30102⟩⟩
def mergeEvent : Nat := 75589
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75582RawTerms
def rightRaw : List Term := Proof.Events021.exact5499RawTerms
def group : MergeGroup := .operator 75582 5499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75582) (leftOrdinal := 1)
    (rightResult := 5499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6651⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75589

namespace LeftMerge75591
def owner : Owner := ⟨.program ⟨214⟩, ⟨30102⟩⟩
def mergeEvent : Nat := 75591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events021.exact5492RawTerms
def group : MergeGroup := .relation 75590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75590) (rhsResult := 5492)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6651⟩⟩) ⟨6597⟩ 5492) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75591

namespace LeftMerge75605
def owner : Owner := ⟨.program ⟨214⟩, ⟨30090⟩⟩
def mergeEvent : Nat := 75605
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30088⟩⟩] } }
def leftRaw : List Term := Proof.Events256.exact65573RawTerms
def rightRaw : List Term := Proof.Events295.exact75599RawTerms
def group : MergeGroup := .operator 65573 75599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65573) (leftOrdinal := 0)
    (rightResult := 75599) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30088⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30088⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75605

namespace LeftMerge75606
def owner : Owner := ⟨.program ⟨214⟩, ⟨30090⟩⟩
def mergeEvent : Nat := 75606
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17007⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30088⟩⟩] } }
def leftRaw : List Term := Proof.Events256.exact65573RawTerms
def rightRaw : List Term := Proof.Events295.exact75599RawTerms
def group : MergeGroup := .operator 65573 75599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65573) (leftOrdinal := 1)
    (rightResult := 75599) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17007⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30088⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17007⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30088⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75606

namespace LeftMerge75608
def owner : Owner := ⟨.program ⟨214⟩, ⟨30090⟩⟩
def mergeEvent : Nat := 75608
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17007⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24788⟩⟩] } }
def rhsRaw : List Term := Proof.Events295.exact75596RawTerms
def group : MergeGroup := .relation 75607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75607) (rhsResult := 75596)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17007⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30088⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30088⟩⟩) ⟨24788⟩ 75596) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24788⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17007⟩⟩], [⟨.program ⟨214⟩, ⟨24788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75608

namespace LeftMerge75622
def owner : Owner := ⟨.program ⟨214⟩, ⟨22767⟩⟩
def mergeEvent : Nat := 75622
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22764⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events295.exact75616RawTerms
def group : MergeGroup := .operator 65387 75616
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 75616) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22764⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22764⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75622

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
