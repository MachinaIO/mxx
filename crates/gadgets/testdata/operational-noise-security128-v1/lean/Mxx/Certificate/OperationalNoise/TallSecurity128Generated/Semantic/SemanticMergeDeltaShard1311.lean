import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge213552
def owner : Owner := ⟨.program ⟨257⟩, ⟨55502⟩⟩
def mergeEvent : Nat := 213552
def frameStart : Nat := 213459
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213547RawTerms
def rightRaw : List Term := Proof.Events834.exact213504RawTerms
def group : MergeGroup := .operator 213547 213504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213547) (leftOrdinal := 1)
    (rightResult := 213504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213552

namespace LeftMerge213554
def owner : Owner := ⟨.program ⟨257⟩, ⟨55502⟩⟩
def mergeEvent : Nat := 213554
def frameStart : Nat := 213459
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54989⟩⟩] } }
def rhsRaw : List Term := Proof.Events833.exact213501RawTerms
def group : MergeGroup := .relation 213553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213553) (rhsResult := 213501)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55499⟩⟩) ⟨54989⟩ 213501) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54989⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨54989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213554

namespace LeftMerge213562
def owner : Owner := ⟨.program ⟨257⟩, ⟨53870⟩⟩
def mergeEvent : Nat := 213562
def frameStart : Nat := 213459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213515RawTerms
def rightRaw : List Term := Proof.Events834.exact213558RawTerms
def group : MergeGroup := .operator 213515 213558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213515) (leftOrdinal := 0)
    (rightResult := 213558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213562

namespace LeftMerge213579
def owner : Owner := ⟨.program ⟨257⟩, ⟨54432⟩⟩
def mergeEvent : Nat := 213579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events834.exact213576RawTerms
def group : MergeGroup := .relation 213578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213578) (rhsResult := 213576)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 213577 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩]⟩) (none) 213576) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213579

namespace LeftMerge213580
def owner : Owner := ⟨.program ⟨257⟩, ⟨54432⟩⟩
def mergeEvent : Nat := 213580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩] } }
def rhsRaw : List Term := Proof.Events834.exact213576RawTerms
def group : MergeGroup := .relation 213578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213578) (rhsResult := 213576)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 213577 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩]⟩) (none) 213576) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213580

namespace LeftMerge213581
def owner : Owner := ⟨.program ⟨257⟩, ⟨54432⟩⟩
def mergeEvent : Nat := 213581
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54989⟩⟩] } }
def rhsRaw : List Term := Proof.Events834.exact213576RawTerms
def group : MergeGroup := .relation 213578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213578) (rhsResult := 213576)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 213577 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩]⟩) (none) 213576) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54989⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨54989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213581

namespace LeftMerge213582
def owner : Owner := ⟨.program ⟨257⟩, ⟨54432⟩⟩
def mergeEvent : Nat := 213582
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events834.exact213576RawTerms
def group : MergeGroup := .relation 213578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213578) (rhsResult := 213576)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 213577 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54429⟩⟩]⟩) (none) 213576) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213582

namespace LeftMerge213587
def owner : Owner := ⟨.program ⟨257⟩, ⟨55501⟩⟩
def mergeEvent : Nat := 213587
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54989⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213583RawTerms
def rightRaw : List Term := Proof.Events833.exact213397RawTerms
def group : MergeGroup := .operator 213583 213397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213583) (leftOrdinal := 2)
    (rightResult := 213397) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54989⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], [⟨.program ⟨257⟩, ⟨54989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213587

namespace LeftMerge213588
def owner : Owner := ⟨.program ⟨257⟩, ⟨55501⟩⟩
def mergeEvent : Nat := 213588
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213583RawTerms
def rightRaw : List Term := Proof.Events833.exact213397RawTerms
def group : MergeGroup := .operator 213583 213397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213583) (leftOrdinal := 1)
    (rightResult := 213397) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55499⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213588

namespace LeftMerge213596
def owner : Owner := ⟨.program ⟨257⟩, ⟨55934⟩⟩
def mergeEvent : Nat := 213596
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55932⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213590RawTerms
def rightRaw : List Term := Proof.Events833.exact213313RawTerms
def group : MergeGroup := .operator 213590 213313
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213590) (leftOrdinal := 0)
    (rightResult := 213313) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55932⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55932⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213596

namespace LeftMerge213597
def owner : Owner := ⟨.program ⟨257⟩, ⟨55934⟩⟩
def mergeEvent : Nat := 213597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55932⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213590RawTerms
def rightRaw : List Term := Proof.Events833.exact213313RawTerms
def group : MergeGroup := .operator 213590 213313
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213590) (leftOrdinal := 1)
    (rightResult := 213313) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55932⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55932⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213597

namespace LeftMerge213599
def owner : Owner := ⟨.program ⟨257⟩, ⟨55934⟩⟩
def mergeEvent : Nat := 213599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55141⟩⟩] } }
def rhsRaw : List Term := Proof.Events833.exact213310RawTerms
def group : MergeGroup := .relation 213598
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213598) (rhsResult := 213310)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55932⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55932⟩⟩) ⟨55141⟩ 213310) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55141⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨55141⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213599

namespace LeftMerge213613
def owner : Owner := ⟨.program ⟨257⟩, ⟨54739⟩⟩
def mergeEvent : Nat := 213613
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54736⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events834.exact213607RawTerms
def group : MergeGroup := .operator 207620 213607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 213607) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54736⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54736⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213613

namespace LeftMerge213734
def owner : Owner := ⟨.program ⟨257⟩, ⟨55348⟩⟩
def mergeEvent : Nat := 213734
def frameStart : Nat := 213668
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213730RawTerms
def rightRaw : List Term := Proof.Events834.exact213728RawTerms
def group : MergeGroup := .operator 213730 213728
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213730) (leftOrdinal := 0)
    (rightResult := 213728) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213734

namespace LeftMerge213746
def owner : Owner := ⟨.program ⟨257⟩, ⟨55933⟩⟩
def mergeEvent : Nat := 213746
def frameStart : Nat := 213668
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55932⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213742RawTerms
def rightRaw : List Term := Proof.Events834.exact213719RawTerms
def group : MergeGroup := .operator 213742 213719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213742) (leftOrdinal := 0)
    (rightResult := 213719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55932⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55932⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213746

namespace LeftMerge213747
def owner : Owner := ⟨.program ⟨257⟩, ⟨55933⟩⟩
def mergeEvent : Nat := 213747
def frameStart : Nat := 213668
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55932⟩⟩] } }
def leftRaw : List Term := Proof.Events834.exact213742RawTerms
def rightRaw : List Term := Proof.Events834.exact213719RawTerms
def group : MergeGroup := .operator 213742 213719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213742) (leftOrdinal := 1)
    (rightResult := 213719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55932⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55932⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213747

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
