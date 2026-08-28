import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge199734
def owner : Owner := ⟨.program ⟨257⟩, ⟨33482⟩⟩
def mergeEvent : Nat := 199734
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32961⟩⟩] } }
def rhsRaw : List Term := Proof.Events779.exact199659RawTerms
def group : MergeGroup := .relation 199733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199733) (rhsResult := 199659)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33481⟩⟩) ⟨32961⟩ 199659) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32961⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], [⟨.program ⟨257⟩, ⟨32961⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199734

namespace LeftMerge199735
def owner : Owner := ⟨.program ⟨257⟩, ⟨33482⟩⟩
def mergeEvent : Nat := 199735
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199726RawTerms
def rightRaw : List Term := Proof.Events779.exact199662RawTerms
def group : MergeGroup := .operator 199726 199662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199726) (leftOrdinal := 0)
    (rightResult := 199662) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33481⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199735

namespace LeftMerge199749
def owner : Owner := ⟨.program ⟨257⟩, ⟨32412⟩⟩
def mergeEvent : Nat := 199749
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events780.exact199743RawTerms
def group : MergeGroup := .operator 192995 199743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 199743) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32409⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199749

namespace LeftMerge199828
def owner : Owner := ⟨.program ⟨257⟩, ⟨31540⟩⟩
def mergeEvent : Nat := 199828
def frameStart : Nat := 199798
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events780.exact199824RawTerms
def rightRaw : List Term := Proof.Events780.exact199821RawTerms
def group : MergeGroup := .operator 199824 199821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199824) (leftOrdinal := 0)
    (rightResult := 199821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24314⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199828

namespace LeftMerge199858
def owner : Owner := ⟨.program ⟨257⟩, ⟨33236⟩⟩
def mergeEvent : Nat := 199858
def frameStart : Nat := 199798
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199854RawTerms
def rightRaw : List Term := Proof.Events780.exact199852RawTerms
def group : MergeGroup := .operator 199854 199852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199854) (leftOrdinal := 0)
    (rightResult := 199852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199858

namespace LeftMerge199881
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 199881
def frameStart : Nat := 199798
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199877RawTerms
def rightRaw : List Term := Proof.Events780.exact199874RawTerms
def group : MergeGroup := .operator 199877 199874
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199877) (leftOrdinal := 0)
    (rightResult := 199874) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199881

namespace LeftMerge199890
def owner : Owner := ⟨.program ⟨257⟩, ⟨33484⟩⟩
def mergeEvent : Nat := 199890
def frameStart : Nat := 199798
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199886RawTerms
def rightRaw : List Term := Proof.Events780.exact199843RawTerms
def group : MergeGroup := .operator 199886 199843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199886) (leftOrdinal := 0)
    (rightResult := 199843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33481⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199890

namespace LeftMerge199891
def owner : Owner := ⟨.program ⟨257⟩, ⟨33484⟩⟩
def mergeEvent : Nat := 199891
def frameStart : Nat := 199798
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199886RawTerms
def rightRaw : List Term := Proof.Events780.exact199843RawTerms
def group : MergeGroup := .operator 199886 199843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199886) (leftOrdinal := 1)
    (rightResult := 199843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33481⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199891

namespace LeftMerge199893
def owner : Owner := ⟨.program ⟨257⟩, ⟨33484⟩⟩
def mergeEvent : Nat := 199893
def frameStart : Nat := 199798
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32961⟩⟩] } }
def rhsRaw : List Term := Proof.Events780.exact199840RawTerms
def group : MergeGroup := .relation 199892
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199892) (rhsResult := 199840)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33481⟩⟩) ⟨32961⟩ 199840) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32961⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], [⟨.program ⟨257⟩, ⟨32961⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199893

namespace LeftMerge199901
def owner : Owner := ⟨.program ⟨257⟩, ⟨31846⟩⟩
def mergeEvent : Nat := 199901
def frameStart : Nat := 199798
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199854RawTerms
def rightRaw : List Term := Proof.Events780.exact199897RawTerms
def group : MergeGroup := .operator 199854 199897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199854) (leftOrdinal := 0)
    (rightResult := 199897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199901

namespace LeftMerge199918
def owner : Owner := ⟨.program ⟨257⟩, ⟨32412⟩⟩
def mergeEvent : Nat := 199918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events780.exact199915RawTerms
def group : MergeGroup := .relation 199917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199917) (rhsResult := 199915)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 199916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩]⟩) (none) 199915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199918

namespace LeftMerge199919
def owner : Owner := ⟨.program ⟨257⟩, ⟨32412⟩⟩
def mergeEvent : Nat := 199919
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩] } }
def rhsRaw : List Term := Proof.Events780.exact199915RawTerms
def group : MergeGroup := .relation 199917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199917) (rhsResult := 199915)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 199916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩]⟩) (none) 199915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199919

namespace LeftMerge199920
def owner : Owner := ⟨.program ⟨257⟩, ⟨32412⟩⟩
def mergeEvent : Nat := 199920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32961⟩⟩] } }
def rhsRaw : List Term := Proof.Events780.exact199915RawTerms
def group : MergeGroup := .relation 199917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199917) (rhsResult := 199915)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 199916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩]⟩) (none) 199915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32961⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], [⟨.program ⟨257⟩, ⟨32961⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199920

namespace LeftMerge199921
def owner : Owner := ⟨.program ⟨257⟩, ⟨32412⟩⟩
def mergeEvent : Nat := 199921
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events780.exact199915RawTerms
def group : MergeGroup := .relation 199917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199917) (rhsResult := 199915)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 199916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32409⟩⟩]⟩) (none) 199915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199921

namespace LeftMerge199926
def owner : Owner := ⟨.program ⟨257⟩, ⟨33483⟩⟩
def mergeEvent : Nat := 199926
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32961⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199922RawTerms
def rightRaw : List Term := Proof.Events780.exact199736RawTerms
def group : MergeGroup := .operator 199922 199736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199922) (leftOrdinal := 2)
    (rightResult := 199736) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32961⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32961⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24314⟩⟩, ⟨.program ⟨257⟩, ⟨31539⟩⟩], [⟨.program ⟨257⟩, ⟨32961⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199926

namespace LeftMerge199927
def owner : Owner := ⟨.program ⟨257⟩, ⟨33483⟩⟩
def mergeEvent : Nat := 199927
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199922RawTerms
def rightRaw : List Term := Proof.Events780.exact199736RawTerms
def group : MergeGroup := .operator 199922 199736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199922) (leftOrdinal := 1)
    (rightResult := 199736) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33481⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199927

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
