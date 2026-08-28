import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge256972
def owner : Owner := ⟨.program ⟨257⟩, ⟨57362⟩⟩
def mergeEvent : Nat := 256972
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events1003.exact256969RawTerms
def group : MergeGroup := .relation 256971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256971) (rhsResult := 256969)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256970 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩]⟩) (none) 256969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256972

namespace LeftMerge256973
def owner : Owner := ⟨.program ⟨257⟩, ⟨57362⟩⟩
def mergeEvent : Nat := 256973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩] } }
def rhsRaw : List Term := Proof.Events1003.exact256969RawTerms
def group : MergeGroup := .relation 256971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256971) (rhsResult := 256969)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256970 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩]⟩) (none) 256969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256973

namespace LeftMerge256974
def owner : Owner := ⟨.program ⟨257⟩, ⟨57362⟩⟩
def mergeEvent : Nat := 256974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57939⟩⟩] } }
def rhsRaw : List Term := Proof.Events1003.exact256969RawTerms
def group : MergeGroup := .relation 256971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256971) (rhsResult := 256969)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256970 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩]⟩) (none) 256969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57939⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨57939⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256974

namespace LeftMerge256975
def owner : Owner := ⟨.program ⟨257⟩, ⟨57362⟩⟩
def mergeEvent : Nat := 256975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1003.exact256969RawTerms
def group : MergeGroup := .relation 256971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256971) (rhsResult := 256969)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256970 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57359⟩⟩]⟩) (none) 256969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256975

namespace LeftMerge256980
def owner : Owner := ⟨.program ⟨257⟩, ⟨58426⟩⟩
def mergeEvent : Nat := 256980
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57939⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256976RawTerms
def rightRaw : List Term := Proof.Events1003.exact256790RawTerms
def group : MergeGroup := .operator 256976 256790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256976) (leftOrdinal := 2)
    (rightResult := 256790) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57939⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57939⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], [⟨.program ⟨257⟩, ⟨57939⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256980

namespace LeftMerge256981
def owner : Owner := ⟨.program ⟨257⟩, ⟨58426⟩⟩
def mergeEvent : Nat := 256981
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256976RawTerms
def rightRaw : List Term := Proof.Events1003.exact256790RawTerms
def group : MergeGroup := .operator 256976 256790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256976) (leftOrdinal := 1)
    (rightResult := 256790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58424⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256981

namespace LeftMerge256989
def owner : Owner := ⟨.program ⟨257⟩, ⟨58759⟩⟩
def mergeEvent : Nat := 256989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256983RawTerms
def rightRaw : List Term := Proof.Events1002.exact256706RawTerms
def group : MergeGroup := .operator 256983 256706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256983) (leftOrdinal := 0)
    (rightResult := 256706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58757⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256989

namespace LeftMerge256990
def owner : Owner := ⟨.program ⟨257⟩, ⟨58759⟩⟩
def mergeEvent : Nat := 256990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩] } }
def leftRaw : List Term := Proof.Events1003.exact256983RawTerms
def rightRaw : List Term := Proof.Events1002.exact256706RawTerms
def group : MergeGroup := .operator 256983 256706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256983) (leftOrdinal := 1)
    (rightResult := 256706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58757⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256990

namespace LeftMerge256992
def owner : Owner := ⟨.program ⟨257⟩, ⟨58759⟩⟩
def mergeEvent : Nat := 256992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58076⟩⟩] } }
def rhsRaw : List Term := Proof.Events1002.exact256703RawTerms
def group : MergeGroup := .relation 256991
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256991) (rhsResult := 256703)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58757⟩⟩) ⟨58076⟩ 256703) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58076⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨56808⟩⟩], [⟨.program ⟨257⟩, ⟨58076⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256992

namespace LeftMerge257006
def owner : Owner := ⟨.program ⟨257⟩, ⟨57619⟩⟩
def mergeEvent : Nat := 257006
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57616⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1003.exact257000RawTerms
def group : MergeGroup := .operator 251495 257000
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 257000) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57616⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257006

namespace LeftMerge257127
def owner : Owner := ⟨.program ⟨257⟩, ⟨58308⟩⟩
def mergeEvent : Nat := 257127
def frameStart : Nat := 257061
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1004.exact257123RawTerms
def rightRaw : List Term := Proof.Events1004.exact257121RawTerms
def group : MergeGroup := .operator 257123 257121
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257123) (leftOrdinal := 0)
    (rightResult := 257121) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257127

namespace LeftMerge257139
def owner : Owner := ⟨.program ⟨257⟩, ⟨58758⟩⟩
def mergeEvent : Nat := 257139
def frameStart : Nat := 257061
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩] } }
def leftRaw : List Term := Proof.Events1004.exact257135RawTerms
def rightRaw : List Term := Proof.Events1004.exact257112RawTerms
def group : MergeGroup := .operator 257135 257112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257135) (leftOrdinal := 0)
    (rightResult := 257112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58757⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257139

namespace LeftMerge257140
def owner : Owner := ⟨.program ⟨257⟩, ⟨58758⟩⟩
def mergeEvent : Nat := 257140
def frameStart : Nat := 257061
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩] } }
def leftRaw : List Term := Proof.Events1004.exact257135RawTerms
def rightRaw : List Term := Proof.Events1004.exact257112RawTerms
def group : MergeGroup := .operator 257135 257112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257135) (leftOrdinal := 1)
    (rightResult := 257112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58757⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257140

namespace LeftMerge257142
def owner : Owner := ⟨.program ⟨257⟩, ⟨58758⟩⟩
def mergeEvent : Nat := 257142
def frameStart : Nat := 257061
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58076⟩⟩] } }
def rhsRaw : List Term := Proof.Events1004.exact257109RawTerms
def group : MergeGroup := .relation 257141
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257141) (rhsResult := 257109)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58757⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58757⟩⟩) ⟨58076⟩ 257109) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58076⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56808⟩⟩], [⟨.program ⟨257⟩, ⟨58076⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257142

namespace LeftMerge257150
def owner : Owner := ⟨.program ⟨257⟩, ⟨57028⟩⟩
def mergeEvent : Nat := 257150
def frameStart : Nat := 257061
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1004.exact257123RawTerms
def rightRaw : List Term := Proof.Events1004.exact257146RawTerms
def group : MergeGroup := .operator 257123 257146
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257123) (leftOrdinal := 0)
    (rightResult := 257146) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57026⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257150

namespace LeftMerge257167
def owner : Owner := ⟨.program ⟨257⟩, ⟨57619⟩⟩
def mergeEvent : Nat := 257167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }
def rhsRaw : List Term := Proof.Events1004.exact257164RawTerms
def group : MergeGroup := .relation 257166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257166) (rhsResult := 257164)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 257165 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57616⟩⟩]⟩) (none) 257164) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257167

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
