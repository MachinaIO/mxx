import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge52676
def owner : Owner := ⟨.program ⟨257⟩, ⟨55590⟩⟩
def mergeEvent : Nat := 52676
def frameStart : Nat := 52584
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩] } }
def leftRaw : List Term := Proof.Events205.exact52672RawTerms
def rightRaw : List Term := Proof.Events205.exact52629RawTerms
def group : MergeGroup := .operator 52672 52629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52672) (leftOrdinal := 0)
    (rightResult := 52629) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55587⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52676

namespace LeftMerge52677
def owner : Owner := ⟨.program ⟨257⟩, ⟨55590⟩⟩
def mergeEvent : Nat := 52677
def frameStart : Nat := 52584
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩] } }
def leftRaw : List Term := Proof.Events205.exact52672RawTerms
def rightRaw : List Term := Proof.Events205.exact52629RawTerms
def group : MergeGroup := .operator 52672 52629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52672) (leftOrdinal := 1)
    (rightResult := 52629) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55587⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52677

namespace LeftMerge52679
def owner : Owner := ⟨.program ⟨257⟩, ⟨55590⟩⟩
def mergeEvent : Nat := 52679
def frameStart : Nat := 52584
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55037⟩⟩] } }
def rhsRaw : List Term := Proof.Events205.exact52626RawTerms
def group : MergeGroup := .relation 52678
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52678) (rhsResult := 52626)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55587⟩⟩) ⟨55037⟩ 52626) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55037⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], [⟨.program ⟨257⟩, ⟨55037⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52679

namespace LeftMerge52687
def owner : Owner := ⟨.program ⟨257⟩, ⟨53934⟩⟩
def mergeEvent : Nat := 52687
def frameStart : Nat := 52584
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53932⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events205.exact52640RawTerms
def rightRaw : List Term := Proof.Events205.exact52683RawTerms
def group : MergeGroup := .operator 52640 52683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52640) (leftOrdinal := 0)
    (rightResult := 52683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53932⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52687

namespace LeftMerge52704
def owner : Owner := ⟨.program ⟨257⟩, ⟨54512⟩⟩
def mergeEvent : Nat := 52704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events205.exact52701RawTerms
def group : MergeGroup := .relation 52703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52703) (rhsResult := 52701)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52702 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54509⟩⟩]⟩) (none) 52701) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52704

namespace LeftMerge52705
def owner : Owner := ⟨.program ⟨257⟩, ⟨54512⟩⟩
def mergeEvent : Nat := 52705
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩] } }
def rhsRaw : List Term := Proof.Events205.exact52701RawTerms
def group : MergeGroup := .relation 52703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52703) (rhsResult := 52701)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52702 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54509⟩⟩]⟩) (none) 52701) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52705

namespace LeftMerge52706
def owner : Owner := ⟨.program ⟨257⟩, ⟨54512⟩⟩
def mergeEvent : Nat := 52706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55037⟩⟩] } }
def rhsRaw : List Term := Proof.Events205.exact52701RawTerms
def group : MergeGroup := .relation 52703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52703) (rhsResult := 52701)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52702 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54509⟩⟩]⟩) (none) 52701) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55037⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], [⟨.program ⟨257⟩, ⟨55037⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52706

namespace LeftMerge52707
def owner : Owner := ⟨.program ⟨257⟩, ⟨54512⟩⟩
def mergeEvent : Nat := 52707
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events205.exact52701RawTerms
def group : MergeGroup := .relation 52703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52703) (rhsResult := 52701)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52702 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54509⟩⟩]⟩) (none) 52701) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53932⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52707

namespace LeftMerge52712
def owner : Owner := ⟨.program ⟨257⟩, ⟨55589⟩⟩
def mergeEvent : Nat := 52712
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55037⟩⟩] } }
def leftRaw : List Term := Proof.Events205.exact52708RawTerms
def rightRaw : List Term := Proof.Events205.exact52522RawTerms
def group : MergeGroup := .operator 52708 52522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52708) (leftOrdinal := 2)
    (rightResult := 52522) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55037⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55037⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], [⟨.program ⟨257⟩, ⟨55037⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52712

namespace LeftMerge52713
def owner : Owner := ⟨.program ⟨257⟩, ⟨55589⟩⟩
def mergeEvent : Nat := 52713
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩] } }
def leftRaw : List Term := Proof.Events205.exact52708RawTerms
def rightRaw : List Term := Proof.Events205.exact52522RawTerms
def group : MergeGroup := .operator 52708 52522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52708) (leftOrdinal := 1)
    (rightResult := 52522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55587⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52713

namespace LeftMerge52721
def owner : Owner := ⟨.program ⟨257⟩, ⟨56182⟩⟩
def mergeEvent : Nat := 52721
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩] } }
def leftRaw : List Term := Proof.Events205.exact52715RawTerms
def rightRaw : List Term := Proof.Events204.exact52438RawTerms
def group : MergeGroup := .operator 52715 52438
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52715) (leftOrdinal := 0)
    (rightResult := 52438) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56180⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52721

namespace LeftMerge52722
def owner : Owner := ⟨.program ⟨257⟩, ⟨56182⟩⟩
def mergeEvent : Nat := 52722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩] } }
def leftRaw : List Term := Proof.Events205.exact52715RawTerms
def rightRaw : List Term := Proof.Events204.exact52438RawTerms
def group : MergeGroup := .operator 52715 52438
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52715) (leftOrdinal := 1)
    (rightResult := 52438) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56180⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52722

namespace LeftMerge52724
def owner : Owner := ⟨.program ⟨257⟩, ⟨56182⟩⟩
def mergeEvent : Nat := 52724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55213⟩⟩] } }
def rhsRaw : List Term := Proof.Events204.exact52435RawTerms
def group : MergeGroup := .relation 52723
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52723) (rhsResult := 52435)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56180⟩⟩) ⟨55213⟩ 52435) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55213⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52724

namespace LeftMerge52738
def owner : Owner := ⟨.program ⟨257⟩, ⟨54899⟩⟩
def mergeEvent : Nat := 52738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54896⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events205.exact52732RawTerms
def group : MergeGroup := .operator 46745 52732
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 52732) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54896⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52738

namespace LeftMerge52859
def owner : Owner := ⟨.program ⟨257⟩, ⟨55380⟩⟩
def mergeEvent : Nat := 52859
def frameStart : Nat := 52793
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53932⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52855RawTerms
def rightRaw : List Term := Proof.Events206.exact52853RawTerms
def group : MergeGroup := .operator 52855 52853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52855) (leftOrdinal := 0)
    (rightResult := 52853) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53932⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52859

namespace LeftMerge52871
def owner : Owner := ⟨.program ⟨257⟩, ⟨56181⟩⟩
def mergeEvent : Nat := 52871
def frameStart : Nat := 52793
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52867RawTerms
def rightRaw : List Term := Proof.Events206.exact52844RawTerms
def group : MergeGroup := .operator 52867 52844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52867) (leftOrdinal := 0)
    (rightResult := 52844) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56180⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52871

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
