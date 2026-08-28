import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge258731
def owner : Owner := ⟨.program ⟨257⟩, ⟨22322⟩⟩
def mergeEvent : Nat := 258731
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1010.exact258725RawTerms
def group : MergeGroup := .operator 251495 258725
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 258725) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22319⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258731

namespace LeftMerge258810
def owner : Owner := ⟨.program ⟨257⟩, ⟨21375⟩⟩
def mergeEvent : Nat := 258810
def frameStart : Nat := 258780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1010.exact258806RawTerms
def rightRaw : List Term := Proof.Events1010.exact258803RawTerms
def group : MergeGroup := .operator 258806 258803
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258806) (leftOrdinal := 0)
    (rightResult := 258803) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21026⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258810

namespace LeftMerge258840
def owner : Owner := ⟨.program ⟨257⟩, ⟨23188⟩⟩
def mergeEvent : Nat := 258840
def frameStart : Nat := 258780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1011.exact258836RawTerms
def rightRaw : List Term := Proof.Events1011.exact258834RawTerms
def group : MergeGroup := .operator 258836 258834
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258836) (leftOrdinal := 0)
    (rightResult := 258834) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258840

namespace LeftMerge258863
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 258863
def frameStart : Nat := 258780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events1011.exact258859RawTerms
def rightRaw : List Term := Proof.Events1011.exact258856RawTerms
def group : MergeGroup := .operator 258859 258856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258859) (leftOrdinal := 0)
    (rightResult := 258856) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258863

namespace LeftMerge258872
def owner : Owner := ⟨.program ⟨257⟩, ⟨23387⟩⟩
def mergeEvent : Nat := 258872
def frameStart : Nat := 258780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩] } }
def leftRaw : List Term := Proof.Events1011.exact258868RawTerms
def rightRaw : List Term := Proof.Events1011.exact258825RawTerms
def group : MergeGroup := .operator 258868 258825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258868) (leftOrdinal := 0)
    (rightResult := 258825) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23384⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258872

namespace LeftMerge258873
def owner : Owner := ⟨.program ⟨257⟩, ⟨23387⟩⟩
def mergeEvent : Nat := 258873
def frameStart : Nat := 258780
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩] } }
def leftRaw : List Term := Proof.Events1011.exact258868RawTerms
def rightRaw : List Term := Proof.Events1011.exact258825RawTerms
def group : MergeGroup := .operator 258868 258825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258868) (leftOrdinal := 1)
    (rightResult := 258825) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23384⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258873

namespace LeftMerge258875
def owner : Owner := ⟨.program ⟨257⟩, ⟨23387⟩⟩
def mergeEvent : Nat := 258875
def frameStart : Nat := 258780
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22899⟩⟩] } }
def rhsRaw : List Term := Proof.Events1011.exact258822RawTerms
def group : MergeGroup := .relation 258874
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258874) (rhsResult := 258822)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23384⟩⟩) ⟨22899⟩ 258822) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22899⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], [⟨.program ⟨257⟩, ⟨22899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258875

namespace LeftMerge258883
def owner : Owner := ⟨.program ⟨257⟩, ⟨21770⟩⟩
def mergeEvent : Nat := 258883
def frameStart : Nat := 258780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1011.exact258836RawTerms
def rightRaw : List Term := Proof.Events1011.exact258879RawTerms
def group : MergeGroup := .operator 258836 258879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258836) (leftOrdinal := 0)
    (rightResult := 258879) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21768⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258883

namespace LeftMerge258900
def owner : Owner := ⟨.program ⟨257⟩, ⟨22322⟩⟩
def mergeEvent : Nat := 258900
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events1011.exact258897RawTerms
def group : MergeGroup := .relation 258899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258899) (rhsResult := 258897)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258898 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩) (none) 258897) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258900

namespace LeftMerge258901
def owner : Owner := ⟨.program ⟨257⟩, ⟨22322⟩⟩
def mergeEvent : Nat := 258901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩] } }
def rhsRaw : List Term := Proof.Events1011.exact258897RawTerms
def group : MergeGroup := .relation 258899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258899) (rhsResult := 258897)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258898 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩) (none) 258897) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258901

namespace LeftMerge258902
def owner : Owner := ⟨.program ⟨257⟩, ⟨22322⟩⟩
def mergeEvent : Nat := 258902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22899⟩⟩] } }
def rhsRaw : List Term := Proof.Events1011.exact258897RawTerms
def group : MergeGroup := .relation 258899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258899) (rhsResult := 258897)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258898 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩) (none) 258897) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22899⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], [⟨.program ⟨257⟩, ⟨22899⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258902

namespace LeftMerge258903
def owner : Owner := ⟨.program ⟨257⟩, ⟨22322⟩⟩
def mergeEvent : Nat := 258903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1011.exact258897RawTerms
def group : MergeGroup := .relation 258899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258899) (rhsResult := 258897)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258898 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩) (none) 258897) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258903

namespace LeftMerge258908
def owner : Owner := ⟨.program ⟨257⟩, ⟨23386⟩⟩
def mergeEvent : Nat := 258908
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22899⟩⟩] } }
def leftRaw : List Term := Proof.Events1011.exact258904RawTerms
def rightRaw : List Term := Proof.Events1010.exact258718RawTerms
def group : MergeGroup := .operator 258904 258718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258904) (leftOrdinal := 2)
    (rightResult := 258718) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22899⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22899⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], [⟨.program ⟨257⟩, ⟨22899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258908

namespace LeftMerge258909
def owner : Owner := ⟨.program ⟨257⟩, ⟨23386⟩⟩
def mergeEvent : Nat := 258909
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩] } }
def leftRaw : List Term := Proof.Events1011.exact258904RawTerms
def rightRaw : List Term := Proof.Events1010.exact258718RawTerms
def group : MergeGroup := .operator 258904 258718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258904) (leftOrdinal := 1)
    (rightResult := 258718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23384⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258909

namespace LeftMerge258917
def owner : Owner := ⟨.program ⟨257⟩, ⟨23719⟩⟩
def mergeEvent : Nat := 258917
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23717⟩⟩] } }
def leftRaw : List Term := Proof.Events1011.exact258911RawTerms
def rightRaw : List Term := Proof.Events1010.exact258634RawTerms
def group : MergeGroup := .operator 258911 258634
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258911) (leftOrdinal := 0)
    (rightResult := 258634) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23717⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23717⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258917

namespace LeftMerge258918
def owner : Owner := ⟨.program ⟨257⟩, ⟨23719⟩⟩
def mergeEvent : Nat := 258918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23717⟩⟩] } }
def leftRaw : List Term := Proof.Events1011.exact258911RawTerms
def rightRaw : List Term := Proof.Events1010.exact258634RawTerms
def group : MergeGroup := .operator 258911 258634
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258911) (leftOrdinal := 1)
    (rightResult := 258634) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23717⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23717⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258918

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
