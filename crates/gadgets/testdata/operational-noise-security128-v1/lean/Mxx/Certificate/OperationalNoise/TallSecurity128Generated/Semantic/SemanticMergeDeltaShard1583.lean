import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge256753
def owner : Owner := ⟨.program ⟨257⟩, ⟨8026⟩⟩
def mergeEvent : Nat := 256753
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251273RawTerms
def rightRaw : List Term := Proof.Events088.exact22632RawTerms
def group : MergeGroup := .operator 251273 22632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251273) (leftOrdinal := 0)
    (rightResult := 22632) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256753

namespace LeftMerge256770
def owner : Owner := ⟨.program ⟨257⟩, ⟨56377⟩⟩
def mergeEvent : Nat := 256770
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events1002.exact256764RawTerms
def rightRaw : List Term := Proof.Events088.exact22621RawTerms
def group : MergeGroup := .operator 256764 22621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256764) (leftOrdinal := 1)
    (rightResult := 22621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256770

namespace LeftMerge256772
def owner : Owner := ⟨.program ⟨257⟩, ⟨56377⟩⟩
def mergeEvent : Nat := 256772
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22591RawTerms
def group : MergeGroup := .relation 256771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256771) (rhsResult := 22591)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9532⟩⟩) ⟨7273⟩ 22591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256772

namespace LeftMerge256773
def owner : Owner := ⟨.program ⟨257⟩, ⟨56377⟩⟩
def mergeEvent : Nat := 256773
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events1002.exact256764RawTerms
def rightRaw : List Term := Proof.Events088.exact22621RawTerms
def group : MergeGroup := .operator 256764 22621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256764) (leftOrdinal := 0)
    (rightResult := 22621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256773

namespace LeftMerge256778
def owner : Owner := ⟨.program ⟨257⟩, ⟨56378⟩⟩
def mergeEvent : Nat := 256778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256774RawTerms
def rightRaw : List Term := Proof.Events1002.exact256744RawTerms
def group : MergeGroup := .operator 256774 256744
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256774) (leftOrdinal := 1)
    (rightResult := 256744) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256778

namespace LeftMerge256786
def owner : Owner := ⟨.program ⟨257⟩, ⟨58425⟩⟩
def mergeEvent : Nat := 256786
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256780RawTerms
def rightRaw : List Term := Proof.Events1002.exact256716RawTerms
def group : MergeGroup := .operator 256780 256716
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256780) (leftOrdinal := 1)
    (rightResult := 256716) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58424⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256786

namespace LeftMerge256788
def owner : Owner := ⟨.program ⟨257⟩, ⟨58425⟩⟩
def mergeEvent : Nat := 256788
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57939⟩⟩] } }
def rhsRaw : List Term := Proof.Events1002.exact256713RawTerms
def group : MergeGroup := .relation 256787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256787) (rhsResult := 256713)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58424⟩⟩) ⟨57939⟩ 256713) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57939⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨57939⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256788

namespace LeftMerge256789
def owner : Owner := ⟨.program ⟨257⟩, ⟨58425⟩⟩
def mergeEvent : Nat := 256789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256780RawTerms
def rightRaw : List Term := Proof.Events1002.exact256716RawTerms
def group : MergeGroup := .operator 256780 256716
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256780) (leftOrdinal := 0)
    (rightResult := 256716) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58424⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256789

namespace LeftMerge256803
def owner : Owner := ⟨.program ⟨257⟩, ⟨57362⟩⟩
def mergeEvent : Nat := 256803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1003.exact256797RawTerms
def group : MergeGroup := .operator 251495 256797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 256797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57359⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256803

namespace LeftMerge256882
def owner : Owner := ⟨.program ⟨257⟩, ⟨56371⟩⟩
def mergeEvent : Nat := 256882
def frameStart : Nat := 256852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1003.exact256878RawTerms
def rightRaw : List Term := Proof.Events1003.exact256875RawTerms
def group : MergeGroup := .operator 256878 256875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256878) (leftOrdinal := 0)
    (rightResult := 256875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24950⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256882

namespace LeftMerge256912
def owner : Owner := ⟨.program ⟨257⟩, ⟨58228⟩⟩
def mergeEvent : Nat := 256912
def frameStart : Nat := 256852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256908RawTerms
def rightRaw : List Term := Proof.Events1003.exact256906RawTerms
def group : MergeGroup := .operator 256908 256906
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256908) (leftOrdinal := 0)
    (rightResult := 256906) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256912

namespace LeftMerge256935
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 256935
def frameStart : Nat := 256852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256931RawTerms
def rightRaw : List Term := Proof.Events1003.exact256928RawTerms
def group : MergeGroup := .operator 256931 256928
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256931) (leftOrdinal := 0)
    (rightResult := 256928) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256935

namespace LeftMerge256944
def owner : Owner := ⟨.program ⟨257⟩, ⟨58427⟩⟩
def mergeEvent : Nat := 256944
def frameStart : Nat := 256852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256940RawTerms
def rightRaw : List Term := Proof.Events1003.exact256897RawTerms
def group : MergeGroup := .operator 256940 256897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256940) (leftOrdinal := 0)
    (rightResult := 256897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58424⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256944

namespace LeftMerge256945
def owner : Owner := ⟨.program ⟨257⟩, ⟨58427⟩⟩
def mergeEvent : Nat := 256945
def frameStart : Nat := 256852
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256940RawTerms
def rightRaw : List Term := Proof.Events1003.exact256897RawTerms
def group : MergeGroup := .operator 256940 256897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256940) (leftOrdinal := 1)
    (rightResult := 256897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58424⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256945

namespace LeftMerge256947
def owner : Owner := ⟨.program ⟨257⟩, ⟨58427⟩⟩
def mergeEvent : Nat := 256947
def frameStart : Nat := 256852
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57939⟩⟩] } }
def rhsRaw : List Term := Proof.Events1003.exact256894RawTerms
def group : MergeGroup := .relation 256946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256946) (rhsResult := 256894)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58424⟩⟩) ⟨57939⟩ 256894) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57939⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨57939⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256947

namespace LeftMerge256955
def owner : Owner := ⟨.program ⟨257⟩, ⟨56810⟩⟩
def mergeEvent : Nat := 256955
def frameStart : Nat := 256852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256908RawTerms
def rightRaw : List Term := Proof.Events1003.exact256951RawTerms
def group : MergeGroup := .operator 256908 256951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256908) (leftOrdinal := 0)
    (rightResult := 256951) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256955

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
