import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge183848
def owner : Owner := ⟨.program ⟨257⟩, ⟨57442⟩⟩
def mergeEvent : Nat := 183848
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58512⟩⟩] } }
def rhsRaw : List Term := Proof.Events718.exact183844RawTerms
def group : MergeGroup := .relation 183846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183846) (rhsResult := 183844)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 183845 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57439⟩⟩]⟩) (none) 183844) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58512⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58512⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183848

namespace LeftMerge183849
def owner : Owner := ⟨.program ⟨257⟩, ⟨57442⟩⟩
def mergeEvent : Nat := 183849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25046⟩⟩, ⟨.program ⟨257⟩, ⟨56586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57987⟩⟩] } }
def rhsRaw : List Term := Proof.Events718.exact183844RawTerms
def group : MergeGroup := .relation 183846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183846) (rhsResult := 183844)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 183845 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57439⟩⟩]⟩) (none) 183844) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25046⟩⟩, ⟨.program ⟨257⟩, ⟨56586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57987⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25046⟩⟩, ⟨.program ⟨257⟩, ⟨56586⟩⟩], [⟨.program ⟨257⟩, ⟨57987⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183849

namespace LeftMerge183850
def owner : Owner := ⟨.program ⟨257⟩, ⟨57442⟩⟩
def mergeEvent : Nat := 183850
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events718.exact183844RawTerms
def group : MergeGroup := .relation 183846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183846) (rhsResult := 183844)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 183845 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57439⟩⟩]⟩) (none) 183844) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183850

namespace LeftMerge183855
def owner : Owner := ⟨.program ⟨257⟩, ⟨58514⟩⟩
def mergeEvent : Nat := 183855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25046⟩⟩, ⟨.program ⟨257⟩, ⟨56586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57987⟩⟩] } }
def leftRaw : List Term := Proof.Events718.exact183851RawTerms
def rightRaw : List Term := Proof.Events717.exact183665RawTerms
def group : MergeGroup := .operator 183851 183665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183851) (leftOrdinal := 2)
    (rightResult := 183665) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25046⟩⟩, ⟨.program ⟨257⟩, ⟨56586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57987⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25046⟩⟩, ⟨.program ⟨257⟩, ⟨56586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57987⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25046⟩⟩, ⟨.program ⟨257⟩, ⟨56586⟩⟩], [⟨.program ⟨257⟩, ⟨57987⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183855

namespace LeftMerge183856
def owner : Owner := ⟨.program ⟨257⟩, ⟨58514⟩⟩
def mergeEvent : Nat := 183856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58512⟩⟩] } }
def leftRaw : List Term := Proof.Events718.exact183851RawTerms
def rightRaw : List Term := Proof.Events717.exact183665RawTerms
def group : MergeGroup := .operator 183851 183665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183851) (leftOrdinal := 1)
    (rightResult := 183665) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58512⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58512⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58512⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183856

namespace LeftMerge183864
def owner : Owner := ⟨.program ⟨257⟩, ⟨59007⟩⟩
def mergeEvent : Nat := 183864
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩] } }
def leftRaw : List Term := Proof.Events718.exact183858RawTerms
def rightRaw : List Term := Proof.Events717.exact183581RawTerms
def group : MergeGroup := .operator 183858 183581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183858) (leftOrdinal := 0)
    (rightResult := 183581) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59005⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183864

namespace LeftMerge183865
def owner : Owner := ⟨.program ⟨257⟩, ⟨59007⟩⟩
def mergeEvent : Nat := 183865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩] } }
def leftRaw : List Term := Proof.Events718.exact183858RawTerms
def rightRaw : List Term := Proof.Events717.exact183581RawTerms
def group : MergeGroup := .operator 183858 183581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183858) (leftOrdinal := 1)
    (rightResult := 183581) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59005⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183865

namespace LeftMerge183867
def owner : Owner := ⟨.program ⟨257⟩, ⟨59007⟩⟩
def mergeEvent : Nat := 183867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58148⟩⟩] } }
def rhsRaw : List Term := Proof.Events717.exact183578RawTerms
def group : MergeGroup := .relation 183866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183866) (rhsResult := 183578)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59005⟩⟩) ⟨58148⟩ 183578) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58148⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], [⟨.program ⟨257⟩, ⟨58148⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183867

namespace LeftMerge183881
def owner : Owner := ⟨.program ⟨257⟩, ⟨57779⟩⟩
def mergeEvent : Nat := 183881
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57776⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events718.exact183875RawTerms
def group : MergeGroup := .operator 178370 183875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 183875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183881

namespace LeftMerge184002
def owner : Owner := ⟨.program ⟨257⟩, ⟨58340⟩⟩
def mergeEvent : Nat := 184002
def frameStart : Nat := 183936
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events718.exact183998RawTerms
def rightRaw : List Term := Proof.Events718.exact183996RawTerms
def group : MergeGroup := .operator 183998 183996
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183998) (leftOrdinal := 0)
    (rightResult := 183996) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184002

namespace LeftMerge184014
def owner : Owner := ⟨.program ⟨257⟩, ⟨59006⟩⟩
def mergeEvent : Nat := 184014
def frameStart : Nat := 183936
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩] } }
def leftRaw : List Term := Proof.Events718.exact184010RawTerms
def rightRaw : List Term := Proof.Events718.exact183987RawTerms
def group : MergeGroup := .operator 184010 183987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184010) (leftOrdinal := 0)
    (rightResult := 183987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59005⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184014

namespace LeftMerge184015
def owner : Owner := ⟨.program ⟨257⟩, ⟨59006⟩⟩
def mergeEvent : Nat := 184015
def frameStart : Nat := 183936
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩] } }
def leftRaw : List Term := Proof.Events718.exact184010RawTerms
def rightRaw : List Term := Proof.Events718.exact183987RawTerms
def group : MergeGroup := .operator 184010 183987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184010) (leftOrdinal := 1)
    (rightResult := 183987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59005⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184015

namespace LeftMerge184017
def owner : Owner := ⟨.program ⟨257⟩, ⟨59006⟩⟩
def mergeEvent : Nat := 184017
def frameStart : Nat := 183936
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58148⟩⟩] } }
def rhsRaw : List Term := Proof.Events718.exact183984RawTerms
def group : MergeGroup := .relation 184016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184016) (rhsResult := 183984)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59005⟩⟩) ⟨58148⟩ 183984) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58148⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56872⟩⟩], [⟨.program ⟨257⟩, ⟨58148⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184017

namespace LeftMerge184025
def owner : Owner := ⟨.program ⟨257⟩, ⟨57180⟩⟩
def mergeEvent : Nat := 184025
def frameStart : Nat := 183936
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events718.exact183998RawTerms
def rightRaw : List Term := Proof.Events718.exact184021RawTerms
def group : MergeGroup := .operator 183998 184021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183998) (leftOrdinal := 0)
    (rightResult := 184021) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57178⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184025

namespace LeftMerge184042
def owner : Owner := ⟨.program ⟨257⟩, ⟨57779⟩⟩
def mergeEvent : Nat := 184042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }
def rhsRaw : List Term := Proof.Events718.exact184039RawTerms
def group : MergeGroup := .relation 184041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184041) (rhsResult := 184039)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57776⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184040 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57776⟩⟩]⟩) (none) 184039) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184042

namespace LeftMerge184043
def owner : Owner := ⟨.program ⟨257⟩, ⟨57779⟩⟩
def mergeEvent : Nat := 184043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩] } }
def rhsRaw : List Term := Proof.Events718.exact184039RawTerms
def group : MergeGroup := .relation 184041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184041) (rhsResult := 184039)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57776⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184040 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57776⟩⟩]⟩) (none) 184039) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184043

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
