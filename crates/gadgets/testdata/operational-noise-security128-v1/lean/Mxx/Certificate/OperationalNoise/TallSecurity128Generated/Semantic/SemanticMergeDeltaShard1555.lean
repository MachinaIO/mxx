import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge251645
def owner : Owner := ⟨.program ⟨257⟩, ⟨49607⟩⟩
def mergeEvent : Nat := 251645
def frameStart : Nat := 251550
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49119⟩⟩] } }
def rhsRaw : List Term := Proof.Events982.exact251592RawTerms
def group : MergeGroup := .relation 251644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251644) (rhsResult := 251592)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49604⟩⟩) ⟨49119⟩ 251592) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49119⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨49119⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251645

namespace LeftMerge251653
def owner : Owner := ⟨.program ⟨257⟩, ⟨48110⟩⟩
def mergeEvent : Nat := 251653
def frameStart : Nat := 251550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251606RawTerms
def rightRaw : List Term := Proof.Events983.exact251649RawTerms
def group : MergeGroup := .operator 251606 251649
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251606) (leftOrdinal := 0)
    (rightResult := 251649) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251653

namespace LeftMerge251670
def owner : Owner := ⟨.program ⟨257⟩, ⟨48542⟩⟩
def mergeEvent : Nat := 251670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events983.exact251667RawTerms
def group : MergeGroup := .relation 251669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251669) (rhsResult := 251667)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 251668 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩]⟩) (none) 251667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251670

namespace LeftMerge251671
def owner : Owner := ⟨.program ⟨257⟩, ⟨48542⟩⟩
def mergeEvent : Nat := 251671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩] } }
def rhsRaw : List Term := Proof.Events983.exact251667RawTerms
def group : MergeGroup := .relation 251669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251669) (rhsResult := 251667)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 251668 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩]⟩) (none) 251667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251671

namespace LeftMerge251672
def owner : Owner := ⟨.program ⟨257⟩, ⟨48542⟩⟩
def mergeEvent : Nat := 251672
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49119⟩⟩] } }
def rhsRaw : List Term := Proof.Events983.exact251667RawTerms
def group : MergeGroup := .relation 251669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251669) (rhsResult := 251667)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 251668 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩]⟩) (none) 251667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49119⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨49119⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251672

namespace LeftMerge251673
def owner : Owner := ⟨.program ⟨257⟩, ⟨48542⟩⟩
def mergeEvent : Nat := 251673
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events983.exact251667RawTerms
def group : MergeGroup := .relation 251669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251669) (rhsResult := 251667)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 251668 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩]⟩) (none) 251667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251673

namespace LeftMerge251678
def owner : Owner := ⟨.program ⟨257⟩, ⟨49606⟩⟩
def mergeEvent : Nat := 251678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49119⟩⟩] } }
def leftRaw : List Term := Proof.Events983.exact251674RawTerms
def rightRaw : List Term := Proof.Events982.exact251477RawTerms
def group : MergeGroup := .operator 251674 251477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251674) (leftOrdinal := 2)
    (rightResult := 251477) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49119⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49119⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨49119⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251678

namespace LeftMerge251679
def owner : Owner := ⟨.program ⟨257⟩, ⟨49606⟩⟩
def mergeEvent : Nat := 251679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩] } }
def leftRaw : List Term := Proof.Events983.exact251674RawTerms
def rightRaw : List Term := Proof.Events982.exact251477RawTerms
def group : MergeGroup := .operator 251674 251477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251674) (leftOrdinal := 1)
    (rightResult := 251477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251679

namespace LeftMerge251687
def owner : Owner := ⟨.program ⟨257⟩, ⟨49906⟩⟩
def mergeEvent : Nat := 251687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩] } }
def leftRaw : List Term := Proof.Events983.exact251681RawTerms
def rightRaw : List Term := Proof.Events981.exact251388RawTerms
def group : MergeGroup := .operator 251681 251388
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251681) (leftOrdinal := 0)
    (rightResult := 251388) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49904⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251687

namespace LeftMerge251688
def owner : Owner := ⟨.program ⟨257⟩, ⟨49906⟩⟩
def mergeEvent : Nat := 251688
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩] } }
def leftRaw : List Term := Proof.Events983.exact251681RawTerms
def rightRaw : List Term := Proof.Events981.exact251388RawTerms
def group : MergeGroup := .operator 251681 251388
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251681) (leftOrdinal := 1)
    (rightResult := 251388) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49904⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251688

namespace LeftMerge251690
def owner : Owner := ⟨.program ⟨257⟩, ⟨49906⟩⟩
def mergeEvent : Nat := 251690
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49256⟩⟩] } }
def rhsRaw : List Term := Proof.Events981.exact251385RawTerms
def group : MergeGroup := .relation 251689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251689) (rhsResult := 251385)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49904⟩⟩) ⟨49256⟩ 251385) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49256⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨48108⟩⟩], [⟨.program ⟨257⟩, ⟨49256⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251690

namespace LeftMerge251704
def owner : Owner := ⟨.program ⟨257⟩, ⟨48799⟩⟩
def mergeEvent : Nat := 251704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48796⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events983.exact251698RawTerms
def group : MergeGroup := .operator 251495 251698
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 251698) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48796⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48796⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251704

namespace LeftMerge251825
def owner : Owner := ⟨.program ⟨257⟩, ⟨49488⟩⟩
def mergeEvent : Nat := 251825
def frameStart : Nat := 251759
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events983.exact251821RawTerms
def rightRaw : List Term := Proof.Events983.exact251819RawTerms
def group : MergeGroup := .operator 251821 251819
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251821) (leftOrdinal := 0)
    (rightResult := 251819) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251825

namespace LeftMerge251837
def owner : Owner := ⟨.program ⟨257⟩, ⟨49905⟩⟩
def mergeEvent : Nat := 251837
def frameStart : Nat := 251759
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩] } }
def leftRaw : List Term := Proof.Events983.exact251833RawTerms
def rightRaw : List Term := Proof.Events983.exact251810RawTerms
def group : MergeGroup := .operator 251833 251810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251833) (leftOrdinal := 0)
    (rightResult := 251810) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49904⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251837

namespace LeftMerge251838
def owner : Owner := ⟨.program ⟨257⟩, ⟨49905⟩⟩
def mergeEvent : Nat := 251838
def frameStart : Nat := 251759
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩] } }
def leftRaw : List Term := Proof.Events983.exact251833RawTerms
def rightRaw : List Term := Proof.Events983.exact251810RawTerms
def group : MergeGroup := .operator 251833 251810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251833) (leftOrdinal := 1)
    (rightResult := 251810) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49904⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251838

namespace LeftMerge251840
def owner : Owner := ⟨.program ⟨257⟩, ⟨49905⟩⟩
def mergeEvent : Nat := 251840
def frameStart : Nat := 251759
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49256⟩⟩] } }
def rhsRaw : List Term := Proof.Events983.exact251807RawTerms
def group : MergeGroup := .relation 251839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251839) (rhsResult := 251807)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49904⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49904⟩⟩) ⟨49256⟩ 251807) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49256⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48108⟩⟩], [⟨.program ⟨257⟩, ⟨49256⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251840

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
