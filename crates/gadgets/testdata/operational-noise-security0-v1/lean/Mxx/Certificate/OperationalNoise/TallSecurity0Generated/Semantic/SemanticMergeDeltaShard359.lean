import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge59468
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 7)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge59468

namespace LeftMerge59469
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59469
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 24)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15989⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59469

namespace LeftMerge59471
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events197.exact50642RawTerms
def group : MergeGroup := .relation 59470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 59470) (rhsResult := 50642)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15989⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 50642) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15989⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59471

namespace LeftMerge59472
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 6)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge59472

namespace LeftMerge59473
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15870⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 23)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15870⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15870⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59473

namespace LeftMerge59475
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15870⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events197.exact50642RawTerms
def group : MergeGroup := .relation 59474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 59474) (rhsResult := 50642)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15870⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 50642) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15870⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59475

namespace LeftMerge59476
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59476
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 5)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge59476

namespace LeftMerge59477
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59477
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15751⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 22)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15751⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15751⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59477

namespace LeftMerge59479
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59479
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15751⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events197.exact50642RawTerms
def group : MergeGroup := .relation 59478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 59478) (rhsResult := 50642)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15751⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 50642) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15751⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59479

namespace LeftMerge59480
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59480
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 4)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge59480

namespace LeftMerge59481
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59481
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 21)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59481

namespace LeftMerge59483
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59483
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events197.exact50642RawTerms
def group : MergeGroup := .relation 59482
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 59482) (rhsResult := 50642)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 50642) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59483

namespace LeftMerge59484
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59484
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 3)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge59484

namespace LeftMerge59485
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59485
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17336⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 31)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17336⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17336⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59485

namespace LeftMerge59487
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59487
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17336⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events197.exact50642RawTerms
def group : MergeGroup := .relation 59486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 59486) (rhsResult := 50642)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17336⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 50642) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17336⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge59487

namespace LeftMerge59488
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def mergeEvent : Nat := 59488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events232.exact59422RawTerms
def rightRaw : List Term := Proof.Events197.exact50645RawTerms
def group : MergeGroup := .operator 59422 50645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 59422) (leftOrdinal := 2)
    (rightResult := 50645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge59488

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
