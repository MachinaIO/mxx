import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge54563
def owner : Owner := ⟨.program ⟨214⟩, ⟨14653⟩⟩
def mergeEvent : Nat := 54563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54557RawTerms
def rightRaw : List Term := Proof.Events009.exact2525RawTerms
def group : MergeGroup := .operator 54557 2525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54557) (leftOrdinal := 1)
    (rightResult := 2525) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54563

namespace LeftMerge54564
def owner : Owner := ⟨.program ⟨214⟩, ⟨14653⟩⟩
def mergeEvent : Nat := 54564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54557RawTerms
def rightRaw : List Term := Proof.Events009.exact2525RawTerms
def group : MergeGroup := .operator 54557 2525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54557) (leftOrdinal := 0)
    (rightResult := 2525) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54564

namespace LeftMerge54569
def owner : Owner := ⟨.program ⟨214⟩, ⟨14654⟩⟩
def mergeEvent : Nat := 54569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2525RawTerms
def rightRaw : List Term := Proof.Events197.exact50670RawTerms
def group : MergeGroup := .operator 2525 50670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2525) (leftOrdinal := 0)
    (rightResult := 50670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54569

namespace LeftMerge54574
def owner : Owner := ⟨.program ⟨214⟩, ⟨7256⟩⟩
def mergeEvent : Nat := 54574
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50540RawTerms
def rightRaw : List Term := Proof.Events041.exact10521RawTerms
def group : MergeGroup := .operator 50540 10521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50540) (leftOrdinal := 0)
    (rightResult := 10521) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54574

namespace LeftMerge54591
def owner : Owner := ⟨.program ⟨214⟩, ⟨14657⟩⟩
def mergeEvent : Nat := 54591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54585RawTerms
def rightRaw : List Term := Proof.Events041.exact10510RawTerms
def group : MergeGroup := .operator 54585 10510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54585) (leftOrdinal := 1)
    (rightResult := 10510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7858⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54591

namespace LeftMerge54593
def owner : Owner := ⟨.program ⟨214⟩, ⟨14657⟩⟩
def mergeEvent : Nat := 54593
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }
def rhsRaw : List Term := Proof.Events040.exact10480RawTerms
def group : MergeGroup := .relation 54592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54592) (rhsResult := 10480)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7858⟩⟩) ⟨6781⟩ 10480) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54593

namespace LeftMerge54594
def owner : Owner := ⟨.program ⟨214⟩, ⟨14657⟩⟩
def mergeEvent : Nat := 54594
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54585RawTerms
def rightRaw : List Term := Proof.Events041.exact10510RawTerms
def group : MergeGroup := .operator 54585 10510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54585) (leftOrdinal := 0)
    (rightResult := 10510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7858⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54594

namespace LeftMerge54599
def owner : Owner := ⟨.program ⟨214⟩, ⟨14658⟩⟩
def mergeEvent : Nat := 54599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54595RawTerms
def rightRaw : List Term := Proof.Events213.exact54565RawTerms
def group : MergeGroup := .operator 54595 54565
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54595) (leftOrdinal := 1)
    (rightResult := 54565) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54599

namespace LeftMerge54607
def owner : Owner := ⟨.program ⟨214⟩, ⟨26226⟩⟩
def mergeEvent : Nat := 54607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54601RawTerms
def rightRaw : List Term := Proof.Events213.exact54537RawTerms
def group : MergeGroup := .operator 54601 54537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54601) (leftOrdinal := 1)
    (rightResult := 54537) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26225⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54607

namespace LeftMerge54609
def owner : Owner := ⟨.program ⟨214⟩, ⟨26226⟩⟩
def mergeEvent : Nat := 54609
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }
def rhsRaw : List Term := Proof.Events213.exact54534RawTerms
def group : MergeGroup := .relation 54608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54608) (rhsResult := 54534)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26225⟩⟩) ⟨23670⟩ 54534) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨23670⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54609

namespace LeftMerge54610
def owner : Owner := ⟨.program ⟨214⟩, ⟨26226⟩⟩
def mergeEvent : Nat := 54610
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54601RawTerms
def rightRaw : List Term := Proof.Events213.exact54537RawTerms
def group : MergeGroup := .operator 54601 54537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54601) (leftOrdinal := 0)
    (rightResult := 54537) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26225⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54610

namespace LeftMerge54624
def owner : Owner := ⟨.program ⟨214⟩, ⟨19679⟩⟩
def mergeEvent : Nat := 54624
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events213.exact54618RawTerms
def group : MergeGroup := .operator 50762 54618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 54618) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19676⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54624

namespace LeftMerge54703
def owner : Owner := ⟨.program ⟨214⟩, ⟨14651⟩⟩
def mergeEvent : Nat := 54703
def frameStart : Nat := 54673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events213.exact54699RawTerms
def rightRaw : List Term := Proof.Events213.exact54696RawTerms
def group : MergeGroup := .operator 54699 54696
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54699) (leftOrdinal := 0)
    (rightResult := 54696) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54703

namespace LeftMerge54733
def owner : Owner := ⟨.program ⟨214⟩, ⟨14754⟩⟩
def mergeEvent : Nat := 54733
def frameStart : Nat := 54673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54729RawTerms
def rightRaw : List Term := Proof.Events213.exact54727RawTerms
def group : MergeGroup := .operator 54729 54727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54729) (leftOrdinal := 0)
    (rightResult := 54727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54733

namespace LeftMerge54756
def owner : Owner := ⟨.program ⟨214⟩, ⟨7860⟩⟩
def mergeEvent : Nat := 54756
def frameStart : Nat := 54673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54752RawTerms
def rightRaw : List Term := Proof.Events213.exact54749RawTerms
def group : MergeGroup := .operator 54752 54749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54752) (leftOrdinal := 0)
    (rightResult := 54749) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7858⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54756

namespace LeftMerge54765
def owner : Owner := ⟨.program ⟨214⟩, ⟨26228⟩⟩
def mergeEvent : Nat := 54765
def frameStart : Nat := 54673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54761RawTerms
def rightRaw : List Term := Proof.Events213.exact54718RawTerms
def group : MergeGroup := .operator 54761 54718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54761) (leftOrdinal := 0)
    (rightResult := 54718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26225⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54765

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
