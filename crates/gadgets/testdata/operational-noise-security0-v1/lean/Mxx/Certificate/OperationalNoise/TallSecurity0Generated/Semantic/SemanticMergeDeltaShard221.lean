import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge36575
def owner : Owner := ⟨.program ⟨214⟩, ⟨7301⟩⟩
def mergeEvent : Nat := 36575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35915RawTerms
def rightRaw : List Term := Proof.Events027.exact7014RawTerms
def group : MergeGroup := .operator 35915 7014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35915) (leftOrdinal := 0)
    (rightResult := 7014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36575

namespace LeftMerge36592
def owner : Owner := ⟨.program ⟨214⟩, ⟨10254⟩⟩
def mergeEvent : Nat := 36592
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36586RawTerms
def rightRaw : List Term := Proof.Events027.exact7003RawTerms
def group : MergeGroup := .operator 36586 7003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36586) (leftOrdinal := 1)
    (rightResult := 7003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36592

namespace LeftMerge36594
def owner : Owner := ⟨.program ⟨214⟩, ⟨10254⟩⟩
def mergeEvent : Nat := 36594
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact6973RawTerms
def group : MergeGroup := .relation 36593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36593) (rhsResult := 6973)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7879⟩⟩) ⟨6789⟩ 6973) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36594

namespace LeftMerge36595
def owner : Owner := ⟨.program ⟨214⟩, ⟨10254⟩⟩
def mergeEvent : Nat := 36595
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36586RawTerms
def rightRaw : List Term := Proof.Events027.exact7003RawTerms
def group : MergeGroup := .operator 36586 7003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36586) (leftOrdinal := 0)
    (rightResult := 7003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36595

namespace LeftMerge36600
def owner : Owner := ⟨.program ⟨214⟩, ⟨13177⟩⟩
def mergeEvent : Nat := 36600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36596RawTerms
def rightRaw : List Term := Proof.Events142.exact36566RawTerms
def group : MergeGroup := .operator 36596 36566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36596) (leftOrdinal := 1)
    (rightResult := 36566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36600

namespace LeftMerge36608
def owner : Owner := ⟨.program ⟨214⟩, ⟨25692⟩⟩
def mergeEvent : Nat := 36608
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36602RawTerms
def rightRaw : List Term := Proof.Events142.exact36538RawTerms
def group : MergeGroup := .operator 36602 36538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36602) (leftOrdinal := 1)
    (rightResult := 36538) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25691⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36608

namespace LeftMerge36610
def owner : Owner := ⟨.program ⟨214⟩, ⟨25692⟩⟩
def mergeEvent : Nat := 36610
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23378⟩⟩] } }
def rhsRaw : List Term := Proof.Events142.exact36535RawTerms
def group : MergeGroup := .relation 36609
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36609) (rhsResult := 36535)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25691⟩⟩) ⟨23378⟩ 36535) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23378⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], [⟨.program ⟨214⟩, ⟨23378⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36610

namespace LeftMerge36611
def owner : Owner := ⟨.program ⟨214⟩, ⟨25692⟩⟩
def mergeEvent : Nat := 36611
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36602RawTerms
def rightRaw : List Term := Proof.Events142.exact36538RawTerms
def group : MergeGroup := .operator 36602 36538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36602) (leftOrdinal := 0)
    (rightResult := 36538) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25691⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36611

namespace LeftMerge36625
def owner : Owner := ⟨.program ⟨214⟩, ⟨20187⟩⟩
def mergeEvent : Nat := 36625
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events143.exact36619RawTerms
def group : MergeGroup := .operator 36137 36619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 36619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20184⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36625

namespace LeftMerge36704
def owner : Owner := ⟨.program ⟨214⟩, ⟨13171⟩⟩
def mergeEvent : Nat := 36704
def frameStart : Nat := 36674
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events143.exact36700RawTerms
def rightRaw : List Term := Proof.Events143.exact36697RawTerms
def group : MergeGroup := .operator 36700 36697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36700) (leftOrdinal := 0)
    (rightResult := 36697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10250⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36704

namespace LeftMerge36734
def owner : Owner := ⟨.program ⟨214⟩, ⟨13260⟩⟩
def mergeEvent : Nat := 36734
def frameStart : Nat := 36674
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36730RawTerms
def rightRaw : List Term := Proof.Events143.exact36728RawTerms
def group : MergeGroup := .operator 36730 36728
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36730) (leftOrdinal := 0)
    (rightResult := 36728) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36734

namespace LeftMerge36757
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def mergeEvent : Nat := 36757
def frameStart : Nat := 36674
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36753RawTerms
def rightRaw : List Term := Proof.Events143.exact36750RawTerms
def group : MergeGroup := .operator 36753 36750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36753) (leftOrdinal := 0)
    (rightResult := 36750) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36757

namespace LeftMerge36766
def owner : Owner := ⟨.program ⟨214⟩, ⟨25694⟩⟩
def mergeEvent : Nat := 36766
def frameStart : Nat := 36674
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36762RawTerms
def rightRaw : List Term := Proof.Events143.exact36719RawTerms
def group : MergeGroup := .operator 36762 36719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36762) (leftOrdinal := 0)
    (rightResult := 36719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25691⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36766

namespace LeftMerge36767
def owner : Owner := ⟨.program ⟨214⟩, ⟨25694⟩⟩
def mergeEvent : Nat := 36767
def frameStart : Nat := 36674
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36762RawTerms
def rightRaw : List Term := Proof.Events143.exact36719RawTerms
def group : MergeGroup := .operator 36762 36719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36762) (leftOrdinal := 1)
    (rightResult := 36719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25691⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36767

namespace LeftMerge36769
def owner : Owner := ⟨.program ⟨214⟩, ⟨25694⟩⟩
def mergeEvent : Nat := 36769
def frameStart : Nat := 36674
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23378⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36716RawTerms
def group : MergeGroup := .relation 36768
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36768) (rhsResult := 36716)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25691⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25691⟩⟩) ⟨23378⟩ 36716) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23378⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10250⟩⟩, ⟨.program ⟨214⟩, ⟨13170⟩⟩], [⟨.program ⟨214⟩, ⟨23378⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36769

namespace LeftMerge36777
def owner : Owner := ⟨.program ⟨214⟩, ⟨16881⟩⟩
def mergeEvent : Nat := 36777
def frameStart : Nat := 36674
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36730RawTerms
def rightRaw : List Term := Proof.Events143.exact36773RawTerms
def group : MergeGroup := .operator 36730 36773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36730) (leftOrdinal := 0)
    (rightResult := 36773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36777

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
