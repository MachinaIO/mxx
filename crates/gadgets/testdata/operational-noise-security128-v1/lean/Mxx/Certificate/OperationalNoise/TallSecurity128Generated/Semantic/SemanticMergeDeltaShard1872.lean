import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge302564
def owner : Owner := ⟨.program ⟨257⟩, ⟨17250⟩⟩
def mergeEvent : Nat := 302564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16789⟩⟩] } }
def rhsRaw : List Term := Proof.Events1181.exact302489RawTerms
def group : MergeGroup := .relation 302563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302563) (rhsResult := 302489)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17249⟩⟩) ⟨16789⟩ 302489) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16789⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], [⟨.program ⟨257⟩, ⟨16789⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302564

namespace LeftMerge302565
def owner : Owner := ⟨.program ⟨257⟩, ⟨17250⟩⟩
def mergeEvent : Nat := 302565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩] } }
def leftRaw : List Term := Proof.Events1181.exact302556RawTerms
def rightRaw : List Term := Proof.Events1181.exact302492RawTerms
def group : MergeGroup := .operator 302556 302492
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302556) (leftOrdinal := 0)
    (rightResult := 302492) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302565

namespace LeftMerge302579
def owner : Owner := ⟨.program ⟨257⟩, ⟨16192⟩⟩
def mergeEvent : Nat := 302579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1181.exact302573RawTerms
def group : MergeGroup := .operator 295195 302573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 302573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16189⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302579

namespace LeftMerge302634
def owner : Owner := ⟨.program ⟨257⟩, ⟨15235⟩⟩
def mergeEvent : Nat := 302634
def frameStart : Nat := 302616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1182.exact302630RawTerms
def rightRaw : List Term := Proof.Events1182.exact302627RawTerms
def group : MergeGroup := .operator 302630 302627
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302630) (leftOrdinal := 0)
    (rightResult := 302627) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12231⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302634

namespace LeftMerge302664
def owner : Owner := ⟨.program ⟨257⟩, ⟨17088⟩⟩
def mergeEvent : Nat := 302664
def frameStart : Nat := 302616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1182.exact302660RawTerms
def rightRaw : List Term := Proof.Events1182.exact302658RawTerms
def group : MergeGroup := .operator 302660 302658
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302660) (leftOrdinal := 0)
    (rightResult := 302658) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302664

namespace LeftMerge302687
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 302687
def frameStart : Nat := 302616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events1182.exact302683RawTerms
def rightRaw : List Term := Proof.Events1182.exact302680RawTerms
def group : MergeGroup := .operator 302683 302680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302683) (leftOrdinal := 0)
    (rightResult := 302680) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302687

namespace LeftMerge302696
def owner : Owner := ⟨.program ⟨257⟩, ⟨17252⟩⟩
def mergeEvent : Nat := 302696
def frameStart : Nat := 302616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩] } }
def leftRaw : List Term := Proof.Events1182.exact302692RawTerms
def rightRaw : List Term := Proof.Events1182.exact302649RawTerms
def group : MergeGroup := .operator 302692 302649
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302692) (leftOrdinal := 0)
    (rightResult := 302649) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17249⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302696

namespace LeftMerge302697
def owner : Owner := ⟨.program ⟨257⟩, ⟨17252⟩⟩
def mergeEvent : Nat := 302697
def frameStart : Nat := 302616
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩] } }
def leftRaw : List Term := Proof.Events1182.exact302692RawTerms
def rightRaw : List Term := Proof.Events1182.exact302649RawTerms
def group : MergeGroup := .operator 302692 302649
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302692) (leftOrdinal := 1)
    (rightResult := 302649) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302697

namespace LeftMerge302699
def owner : Owner := ⟨.program ⟨257⟩, ⟨17252⟩⟩
def mergeEvent : Nat := 302699
def frameStart : Nat := 302616
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16789⟩⟩] } }
def rhsRaw : List Term := Proof.Events1182.exact302646RawTerms
def group : MergeGroup := .relation 302698
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302698) (rhsResult := 302646)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17249⟩⟩) ⟨16789⟩ 302646) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16789⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], [⟨.program ⟨257⟩, ⟨16789⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302699

namespace LeftMerge302707
def owner : Owner := ⟨.program ⟨257⟩, ⟨15710⟩⟩
def mergeEvent : Nat := 302707
def frameStart : Nat := 302616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1182.exact302660RawTerms
def rightRaw : List Term := Proof.Events1182.exact302703RawTerms
def group : MergeGroup := .operator 302660 302703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302660) (leftOrdinal := 0)
    (rightResult := 302703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15708⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302707

namespace LeftMerge302724
def owner : Owner := ⟨.program ⟨257⟩, ⟨16192⟩⟩
def mergeEvent : Nat := 302724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events1182.exact302721RawTerms
def group : MergeGroup := .relation 302723
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302723) (rhsResult := 302721)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 302722 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩]⟩) (none) 302721) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302724

namespace LeftMerge302725
def owner : Owner := ⟨.program ⟨257⟩, ⟨16192⟩⟩
def mergeEvent : Nat := 302725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩] } }
def rhsRaw : List Term := Proof.Events1182.exact302721RawTerms
def group : MergeGroup := .relation 302723
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302723) (rhsResult := 302721)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 302722 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩]⟩) (none) 302721) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302725

namespace LeftMerge302726
def owner : Owner := ⟨.program ⟨257⟩, ⟨16192⟩⟩
def mergeEvent : Nat := 302726
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16789⟩⟩] } }
def rhsRaw : List Term := Proof.Events1182.exact302721RawTerms
def group : MergeGroup := .relation 302723
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302723) (rhsResult := 302721)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 302722 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩]⟩) (none) 302721) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16789⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], [⟨.program ⟨257⟩, ⟨16789⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302726

namespace LeftMerge302727
def owner : Owner := ⟨.program ⟨257⟩, ⟨16192⟩⟩
def mergeEvent : Nat := 302727
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1182.exact302721RawTerms
def group : MergeGroup := .relation 302723
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302723) (rhsResult := 302721)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 302722 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16189⟩⟩]⟩) (none) 302721) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302727

namespace LeftMerge302732
def owner : Owner := ⟨.program ⟨257⟩, ⟨17251⟩⟩
def mergeEvent : Nat := 302732
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16789⟩⟩] } }
def leftRaw : List Term := Proof.Events1182.exact302728RawTerms
def rightRaw : List Term := Proof.Events1181.exact302566RawTerms
def group : MergeGroup := .operator 302728 302566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302728) (leftOrdinal := 2)
    (rightResult := 302566) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16789⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16789⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], [⟨.program ⟨257⟩, ⟨16789⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302732

namespace LeftMerge302733
def owner : Owner := ⟨.program ⟨257⟩, ⟨17251⟩⟩
def mergeEvent : Nat := 302733
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩] } }
def leftRaw : List Term := Proof.Events1182.exact302728RawTerms
def rightRaw : List Term := Proof.Events1181.exact302566RawTerms
def group : MergeGroup := .operator 302728 302566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302728) (leftOrdinal := 1)
    (rightResult := 302566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302733

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
