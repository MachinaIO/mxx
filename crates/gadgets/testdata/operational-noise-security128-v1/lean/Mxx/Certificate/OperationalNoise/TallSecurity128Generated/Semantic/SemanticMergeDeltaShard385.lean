import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge65856
def owner : Owner := ⟨.program ⟨257⟩, ⟨64519⟩⟩
def mergeEvent : Nat := 65856
def frameStart : Nat := 65763
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact65851RawTerms
def rightRaw : List Term := Proof.Events257.exact65808RawTerms
def group : MergeGroup := .operator 65851 65808
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65851) (leftOrdinal := 1)
    (rightResult := 65808) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64516⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65856

namespace LeftMerge65858
def owner : Owner := ⟨.program ⟨257⟩, ⟨64519⟩⟩
def mergeEvent : Nat := 65858
def frameStart : Nat := 65763
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63971⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact65805RawTerms
def group : MergeGroup := .relation 65857
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65857) (rhsResult := 65805)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64516⟩⟩) ⟨63971⟩ 65805) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63971⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], [⟨.program ⟨257⟩, ⟨63971⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65858

namespace LeftMerge65866
def owner : Owner := ⟨.program ⟨257⟩, ⟨62866⟩⟩
def mergeEvent : Nat := 65866
def frameStart : Nat := 65763
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact65819RawTerms
def rightRaw : List Term := Proof.Events257.exact65862RawTerms
def group : MergeGroup := .operator 65819 65862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65819) (leftOrdinal := 0)
    (rightResult := 65862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65866

namespace LeftMerge65883
def owner : Owner := ⟨.program ⟨257⟩, ⟨63442⟩⟩
def mergeEvent : Nat := 65883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact65880RawTerms
def group : MergeGroup := .relation 65882
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65882) (rhsResult := 65880)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 65881 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63439⟩⟩]⟩) (none) 65880) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65883

namespace LeftMerge65884
def owner : Owner := ⟨.program ⟨257⟩, ⟨63442⟩⟩
def mergeEvent : Nat := 65884
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact65880RawTerms
def group : MergeGroup := .relation 65882
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65882) (rhsResult := 65880)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 65881 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63439⟩⟩]⟩) (none) 65880) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65884

namespace LeftMerge65885
def owner : Owner := ⟨.program ⟨257⟩, ⟨63442⟩⟩
def mergeEvent : Nat := 65885
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63971⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact65880RawTerms
def group : MergeGroup := .relation 65882
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65882) (rhsResult := 65880)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 65881 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63439⟩⟩]⟩) (none) 65880) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63971⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], [⟨.program ⟨257⟩, ⟨63971⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65885

namespace LeftMerge65886
def owner : Owner := ⟨.program ⟨257⟩, ⟨63442⟩⟩
def mergeEvent : Nat := 65886
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events257.exact65880RawTerms
def group : MergeGroup := .relation 65882
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65882) (rhsResult := 65880)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 65881 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63439⟩⟩]⟩) (none) 65880) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65886

namespace LeftMerge65891
def owner : Owner := ⟨.program ⟨257⟩, ⟨64518⟩⟩
def mergeEvent : Nat := 65891
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63971⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact65887RawTerms
def rightRaw : List Term := Proof.Events256.exact65701RawTerms
def group : MergeGroup := .operator 65887 65701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65887) (leftOrdinal := 2)
    (rightResult := 65701) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63971⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63971⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], [⟨.program ⟨257⟩, ⟨63971⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65891

namespace LeftMerge65892
def owner : Owner := ⟨.program ⟨257⟩, ⟨64518⟩⟩
def mergeEvent : Nat := 65892
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact65887RawTerms
def rightRaw : List Term := Proof.Events256.exact65701RawTerms
def group : MergeGroup := .operator 65887 65701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65887) (leftOrdinal := 1)
    (rightResult := 65701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64516⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65892

namespace LeftMerge65900
def owner : Owner := ⟨.program ⟨257⟩, ⟨65091⟩⟩
def mergeEvent : Nat := 65900
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65089⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact65894RawTerms
def rightRaw : List Term := Proof.Events256.exact65617RawTerms
def group : MergeGroup := .operator 65894 65617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65894) (leftOrdinal := 0)
    (rightResult := 65617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65089⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65089⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65900

namespace LeftMerge65901
def owner : Owner := ⟨.program ⟨257⟩, ⟨65091⟩⟩
def mergeEvent : Nat := 65901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65089⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact65894RawTerms
def rightRaw : List Term := Proof.Events256.exact65617RawTerms
def group : MergeGroup := .operator 65894 65617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65894) (leftOrdinal := 1)
    (rightResult := 65617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65089⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65089⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65901

namespace LeftMerge65903
def owner : Owner := ⟨.program ⟨257⟩, ⟨65091⟩⟩
def mergeEvent : Nat := 65903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64144⟩⟩] } }
def rhsRaw : List Term := Proof.Events256.exact65614RawTerms
def group : MergeGroup := .relation 65902
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65902) (rhsResult := 65614)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65089⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨65089⟩⟩) ⟨64144⟩ 65614) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64144⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨62864⟩⟩], [⟨.program ⟨257⟩, ⟨64144⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65903

namespace LeftMerge65917
def owner : Owner := ⟨.program ⟨257⟩, ⟨63819⟩⟩
def mergeEvent : Nat := 65917
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63816⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events257.exact65911RawTerms
def group : MergeGroup := .operator 61370 65911
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 65911) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63816⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63816⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65917

namespace LeftMerge66038
def owner : Owner := ⟨.program ⟨257⟩, ⟨64316⟩⟩
def mergeEvent : Nat := 66038
def frameStart : Nat := 65972
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact66034RawTerms
def rightRaw : List Term := Proof.Events257.exact66032RawTerms
def group : MergeGroup := .operator 66034 66032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66034) (leftOrdinal := 0)
    (rightResult := 66032) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66038

namespace LeftMerge66050
def owner : Owner := ⟨.program ⟨257⟩, ⟨65090⟩⟩
def mergeEvent : Nat := 66050
def frameStart : Nat := 65972
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65089⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact66046RawTerms
def rightRaw : List Term := Proof.Events257.exact66023RawTerms
def group : MergeGroup := .operator 66046 66023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66046) (leftOrdinal := 0)
    (rightResult := 66023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65089⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65089⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66050

namespace LeftMerge66051
def owner : Owner := ⟨.program ⟨257⟩, ⟨65090⟩⟩
def mergeEvent : Nat := 66051
def frameStart : Nat := 65972
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65089⟩⟩] } }
def leftRaw : List Term := Proof.Events257.exact66046RawTerms
def rightRaw : List Term := Proof.Events257.exact66023RawTerms
def group : MergeGroup := .operator 66046 66023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66046) (leftOrdinal := 1)
    (rightResult := 66023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65089⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65089⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66051

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
