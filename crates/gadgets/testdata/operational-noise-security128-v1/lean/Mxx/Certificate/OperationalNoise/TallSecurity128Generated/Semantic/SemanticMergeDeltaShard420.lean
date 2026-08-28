import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge71562
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 0)
    (rightResult := 70108) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71562

namespace LeftMerge71563
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 19)
    (rightResult := 70108) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71563

namespace LeftMerge71571
def owner : Owner := ⟨.program ⟨257⟩, ⟨71473⟩⟩
def mergeEvent : Nat := 71571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71565RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 71565 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71565) (leftOrdinal := 0)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71571

namespace LeftMerge71572
def owner : Owner := ⟨.program ⟨257⟩, ⟨71473⟩⟩
def mergeEvent : Nat := 71572
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71565RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 71565 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71565) (leftOrdinal := 1)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71572

namespace LeftMerge71574
def owner : Owner := ⟨.program ⟨257⟩, ⟨71473⟩⟩
def mergeEvent : Nat := 71574
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15515RawTerms
def group : MergeGroup := .relation 71573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71573) (rhsResult := 15515)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7139⟩⟩) ⟨7035⟩ 15515) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71574

namespace LeftMerge71588
def owner : Owner := ⟨.program ⟨257⟩, ⟨50200⟩⟩
def mergeEvent : Nat := 71588
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61556RawTerms
def rightRaw : List Term := Proof.Events279.exact71582RawTerms
def group : MergeGroup := .operator 61556 71582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61556) (leftOrdinal := 0)
    (rightResult := 71582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50198⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71588

namespace LeftMerge71589
def owner : Owner := ⟨.program ⟨257⟩, ⟨50200⟩⟩
def mergeEvent : Nat := 71589
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61556RawTerms
def rightRaw : List Term := Proof.Events279.exact71582RawTerms
def group : MergeGroup := .operator 61556 71582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61556) (leftOrdinal := 1)
    (rightResult := 71582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50198⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71589

namespace LeftMerge71591
def owner : Owner := ⟨.program ⟨257⟩, ⟨50200⟩⟩
def mergeEvent : Nat := 71591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49363⟩⟩] } }
def rhsRaw : List Term := Proof.Events279.exact71579RawTerms
def group : MergeGroup := .relation 71590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71590) (rhsResult := 71579)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50198⟩⟩) ⟨49363⟩ 71579) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49363⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71591

namespace LeftMerge71605
def owner : Owner := ⟨.program ⟨257⟩, ⟨49035⟩⟩
def mergeEvent : Nat := 71605
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events279.exact71599RawTerms
def group : MergeGroup := .operator 61370 71599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 71599) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49032⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71605

namespace LeftMerge71726
def owner : Owner := ⟨.program ⟨257⟩, ⟨49536⟩⟩
def mergeEvent : Nat := 71726
def frameStart : Nat := 71660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71722RawTerms
def rightRaw : List Term := Proof.Events280.exact71720RawTerms
def group : MergeGroup := .operator 71722 71720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71722) (leftOrdinal := 0)
    (rightResult := 71720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71726

namespace LeftMerge71738
def owner : Owner := ⟨.program ⟨257⟩, ⟨50199⟩⟩
def mergeEvent : Nat := 71738
def frameStart : Nat := 71660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71734RawTerms
def rightRaw : List Term := Proof.Events280.exact71711RawTerms
def group : MergeGroup := .operator 71734 71711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71734) (leftOrdinal := 0)
    (rightResult := 71711) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50198⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71738

namespace LeftMerge71739
def owner : Owner := ⟨.program ⟨257⟩, ⟨50199⟩⟩
def mergeEvent : Nat := 71739
def frameStart : Nat := 71660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71734RawTerms
def rightRaw : List Term := Proof.Events280.exact71711RawTerms
def group : MergeGroup := .operator 71734 71711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71734) (leftOrdinal := 1)
    (rightResult := 71711) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50198⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71739

namespace LeftMerge71741
def owner : Owner := ⟨.program ⟨257⟩, ⟨50199⟩⟩
def mergeEvent : Nat := 71741
def frameStart : Nat := 71660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49363⟩⟩] } }
def rhsRaw : List Term := Proof.Events280.exact71708RawTerms
def group : MergeGroup := .relation 71740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71740) (rhsResult := 71708)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50198⟩⟩) ⟨49363⟩ 71708) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49363⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71741

namespace LeftMerge71749
def owner : Owner := ⟨.program ⟨257⟩, ⟨48452⟩⟩
def mergeEvent : Nat := 71749
def frameStart : Nat := 71660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71722RawTerms
def rightRaw : List Term := Proof.Events280.exact71745RawTerms
def group : MergeGroup := .operator 71722 71745
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71722) (leftOrdinal := 0)
    (rightResult := 71745) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48450⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71749

namespace LeftMerge71766
def owner : Owner := ⟨.program ⟨257⟩, ⟨49035⟩⟩
def mergeEvent : Nat := 71766
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7231⟩⟩] } }
def rhsRaw : List Term := Proof.Events280.exact71763RawTerms
def group : MergeGroup := .relation 71765
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71765) (rhsResult := 71763)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 71764 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩) (none) 71763) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7231⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71766

namespace LeftMerge71767
def owner : Owner := ⟨.program ⟨257⟩, ⟨49035⟩⟩
def mergeEvent : Nat := 71767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩] } }
def rhsRaw : List Term := Proof.Events280.exact71763RawTerms
def group : MergeGroup := .relation 71765
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71765) (rhsResult := 71763)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 71764 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩) (none) 71763) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71767

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
