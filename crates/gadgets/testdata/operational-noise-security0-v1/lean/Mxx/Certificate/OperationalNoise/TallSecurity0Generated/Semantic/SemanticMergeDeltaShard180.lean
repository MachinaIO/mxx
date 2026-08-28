import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge31570
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31570
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17129⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31366RawTerms
def group : MergeGroup := .relation 31569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31569) (rhsResult := 31366)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 31366) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31570

namespace LeftMerge31571
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31571
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16317⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 26)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16317⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31571

namespace LeftMerge31573
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31573
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16317⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31366RawTerms
def group : MergeGroup := .relation 31572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31572) (rhsResult := 31366)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 31366) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31573

namespace LeftMerge31574
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31574
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18379⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 35)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18379⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31574

namespace LeftMerge31576
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31576
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18379⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31366RawTerms
def group : MergeGroup := .relation 31575
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31575) (rhsResult := 31366)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 31366) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31576

namespace LeftMerge31577
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31577
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16114⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 25)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16114⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31577

namespace LeftMerge31579
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31579
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16114⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31366RawTerms
def group : MergeGroup := .relation 31578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31578) (rhsResult := 31366)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 31366) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31579

namespace LeftMerge31580
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31580
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15995⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 24)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15995⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31580

namespace LeftMerge31582
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31582
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15995⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31366RawTerms
def group : MergeGroup := .relation 31581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31581) (rhsResult := 31366)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 31366) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31582

namespace LeftMerge31583
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31583
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15876⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 23)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15876⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31583

namespace LeftMerge31585
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31585
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15876⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31366RawTerms
def group : MergeGroup := .relation 31584
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31584) (rhsResult := 31366)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 31366) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31585

namespace LeftMerge31586
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31586
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15757⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 22)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15757⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31586

namespace LeftMerge31588
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31588
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15757⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31366RawTerms
def group : MergeGroup := .relation 31587
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31587) (rhsResult := 31366)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 31366) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31588

namespace LeftMerge31589
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31589
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15638⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 21)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15638⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31589

namespace LeftMerge31591
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31591
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15638⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events122.exact31366RawTerms
def group : MergeGroup := .relation 31590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 31590) (rhsResult := 31366)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 31366) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31591

namespace LeftMerge31592
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31592
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 31)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge31592

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
