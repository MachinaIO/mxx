import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge53499
def owner : Owner := ⟨.program ⟨257⟩, ⟨32472⟩⟩
def mergeEvent : Nat := 53499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events208.exact53493RawTerms
def group : MergeGroup := .operator 46745 53493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 53493) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53499

namespace LeftMerge53578
def owner : Owner := ⟨.program ⟨257⟩, ⟨31702⟩⟩
def mergeEvent : Nat := 53578
def frameStart : Nat := 53548
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events209.exact53574RawTerms
def rightRaw : List Term := Proof.Events209.exact53571RawTerms
def group : MergeGroup := .operator 53574 53571
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53574) (leftOrdinal := 0)
    (rightResult := 53571) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24386⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53578

namespace LeftMerge53608
def owner : Owner := ⟨.program ⟨257⟩, ⟨33260⟩⟩
def mergeEvent : Nat := 53608
def frameStart : Nat := 53548
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53604RawTerms
def rightRaw : List Term := Proof.Events209.exact53602RawTerms
def group : MergeGroup := .operator 53604 53602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53604) (leftOrdinal := 0)
    (rightResult := 53602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53608

namespace LeftMerge53631
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 53631
def frameStart : Nat := 53548
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53627RawTerms
def rightRaw : List Term := Proof.Events209.exact53624RawTerms
def group : MergeGroup := .operator 53627 53624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53627) (leftOrdinal := 0)
    (rightResult := 53624) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53631

namespace LeftMerge53640
def owner : Owner := ⟨.program ⟨257⟩, ⟨33550⟩⟩
def mergeEvent : Nat := 53640
def frameStart : Nat := 53548
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53636RawTerms
def rightRaw : List Term := Proof.Events209.exact53593RawTerms
def group : MergeGroup := .operator 53636 53593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53636) (leftOrdinal := 0)
    (rightResult := 53593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53640

namespace LeftMerge53641
def owner : Owner := ⟨.program ⟨257⟩, ⟨33550⟩⟩
def mergeEvent : Nat := 53641
def frameStart : Nat := 53548
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53636RawTerms
def rightRaw : List Term := Proof.Events209.exact53593RawTerms
def group : MergeGroup := .operator 53636 53593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53636) (leftOrdinal := 1)
    (rightResult := 53593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33547⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53641

namespace LeftMerge53643
def owner : Owner := ⟨.program ⟨257⟩, ⟨33550⟩⟩
def mergeEvent : Nat := 53643
def frameStart : Nat := 53548
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32997⟩⟩] } }
def rhsRaw : List Term := Proof.Events209.exact53590RawTerms
def group : MergeGroup := .relation 53642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53642) (rhsResult := 53590)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33547⟩⟩) ⟨32997⟩ 53590) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32997⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], [⟨.program ⟨257⟩, ⟨32997⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53643

namespace LeftMerge53651
def owner : Owner := ⟨.program ⟨257⟩, ⟨31894⟩⟩
def mergeEvent : Nat := 53651
def frameStart : Nat := 53548
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53604RawTerms
def rightRaw : List Term := Proof.Events209.exact53647RawTerms
def group : MergeGroup := .operator 53604 53647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53604) (leftOrdinal := 0)
    (rightResult := 53647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31892⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53651

namespace LeftMerge53668
def owner : Owner := ⟨.program ⟨257⟩, ⟨32472⟩⟩
def mergeEvent : Nat := 53668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events209.exact53665RawTerms
def group : MergeGroup := .relation 53667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53667) (rhsResult := 53665)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53666 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩) (none) 53665) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53668

namespace LeftMerge53669
def owner : Owner := ⟨.program ⟨257⟩, ⟨32472⟩⟩
def mergeEvent : Nat := 53669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩] } }
def rhsRaw : List Term := Proof.Events209.exact53665RawTerms
def group : MergeGroup := .relation 53667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53667) (rhsResult := 53665)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53666 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩) (none) 53665) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53669

namespace LeftMerge53670
def owner : Owner := ⟨.program ⟨257⟩, ⟨32472⟩⟩
def mergeEvent : Nat := 53670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32997⟩⟩] } }
def rhsRaw : List Term := Proof.Events209.exact53665RawTerms
def group : MergeGroup := .relation 53667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53667) (rhsResult := 53665)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53666 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩) (none) 53665) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32997⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], [⟨.program ⟨257⟩, ⟨32997⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53670

namespace LeftMerge53671
def owner : Owner := ⟨.program ⟨257⟩, ⟨32472⟩⟩
def mergeEvent : Nat := 53671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨31892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events209.exact53665RawTerms
def group : MergeGroup := .relation 53667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53667) (rhsResult := 53665)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53666 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩) (none) 53665) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨31892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53671

namespace LeftMerge53676
def owner : Owner := ⟨.program ⟨257⟩, ⟨33549⟩⟩
def mergeEvent : Nat := 53676
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32997⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53672RawTerms
def rightRaw : List Term := Proof.Events208.exact53486RawTerms
def group : MergeGroup := .operator 53672 53486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53672) (leftOrdinal := 2)
    (rightResult := 53486) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32997⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32997⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24386⟩⟩, ⟨.program ⟨257⟩, ⟨31701⟩⟩], [⟨.program ⟨257⟩, ⟨32997⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53676

namespace LeftMerge53677
def owner : Owner := ⟨.program ⟨257⟩, ⟨33549⟩⟩
def mergeEvent : Nat := 53677
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53672RawTerms
def rightRaw : List Term := Proof.Events208.exact53486RawTerms
def group : MergeGroup := .operator 53672 53486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53672) (leftOrdinal := 1)
    (rightResult := 53486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53677

namespace LeftMerge53685
def owner : Owner := ⟨.program ⟨257⟩, ⟨34142⟩⟩
def mergeEvent : Nat := 53685
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34140⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53679RawTerms
def rightRaw : List Term := Proof.Events208.exact53402RawTerms
def group : MergeGroup := .operator 53679 53402
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53679) (leftOrdinal := 0)
    (rightResult := 53402) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34140⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34140⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53685

namespace LeftMerge53686
def owner : Owner := ⟨.program ⟨257⟩, ⟨34142⟩⟩
def mergeEvent : Nat := 53686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨31892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34140⟩⟩] } }
def leftRaw : List Term := Proof.Events209.exact53679RawTerms
def rightRaw : List Term := Proof.Events208.exact53402RawTerms
def group : MergeGroup := .operator 53679 53402
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53679) (leftOrdinal := 1)
    (rightResult := 53402) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨31892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34140⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨31892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34140⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53686

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
