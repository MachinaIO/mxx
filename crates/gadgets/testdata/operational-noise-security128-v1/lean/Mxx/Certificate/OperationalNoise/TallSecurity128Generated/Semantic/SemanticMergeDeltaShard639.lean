import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge105718
def owner : Owner := ⟨.program ⟨257⟩, ⟨46991⟩⟩
def mergeEvent : Nat := 105718
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46475⟩⟩] } }
def rhsRaw : List Term := Proof.Events412.exact105643RawTerms
def group : MergeGroup := .relation 105717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105717) (rhsResult := 105643)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46990⟩⟩) ⟨46475⟩ 105643) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46475⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105718

namespace LeftMerge105719
def owner : Owner := ⟨.program ⟨257⟩, ⟨46991⟩⟩
def mergeEvent : Nat := 105719
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩] } }
def leftRaw : List Term := Proof.Events412.exact105710RawTerms
def rightRaw : List Term := Proof.Events412.exact105646RawTerms
def group : MergeGroup := .operator 105710 105646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105710) (leftOrdinal := 0)
    (rightResult := 105646) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46990⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105719

namespace LeftMerge105733
def owner : Owner := ⟨.program ⟨257⟩, ⟨45922⟩⟩
def mergeEvent : Nat := 105733
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events412.exact105727RawTerms
def group : MergeGroup := .operator 105245 105727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 105727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105733

namespace LeftMerge105812
def owner : Owner := ⟨.program ⟨257⟩, ⟨45179⟩⟩
def mergeEvent : Nat := 105812
def frameStart : Nat := 105782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events413.exact105808RawTerms
def rightRaw : List Term := Proof.Events413.exact105805RawTerms
def group : MergeGroup := .operator 105808 105805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105808) (leftOrdinal := 0)
    (rightResult := 105805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14796⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105812

namespace LeftMerge105842
def owner : Owner := ⟨.program ⟨257⟩, ⟨46752⟩⟩
def mergeEvent : Nat := 105842
def frameStart : Nat := 105782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events413.exact105838RawTerms
def rightRaw : List Term := Proof.Events413.exact105836RawTerms
def group : MergeGroup := .operator 105838 105836
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105838) (leftOrdinal := 0)
    (rightResult := 105836) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105842

namespace LeftMerge105865
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 105865
def frameStart : Nat := 105782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events413.exact105861RawTerms
def rightRaw : List Term := Proof.Events413.exact105858RawTerms
def group : MergeGroup := .operator 105861 105858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105861) (leftOrdinal := 0)
    (rightResult := 105858) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105865

namespace LeftMerge105874
def owner : Owner := ⟨.program ⟨257⟩, ⟨46993⟩⟩
def mergeEvent : Nat := 105874
def frameStart : Nat := 105782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩] } }
def leftRaw : List Term := Proof.Events413.exact105870RawTerms
def rightRaw : List Term := Proof.Events413.exact105827RawTerms
def group : MergeGroup := .operator 105870 105827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105870) (leftOrdinal := 0)
    (rightResult := 105827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46990⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105874

namespace LeftMerge105875
def owner : Owner := ⟨.program ⟨257⟩, ⟨46993⟩⟩
def mergeEvent : Nat := 105875
def frameStart : Nat := 105782
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩] } }
def leftRaw : List Term := Proof.Events413.exact105870RawTerms
def rightRaw : List Term := Proof.Events413.exact105827RawTerms
def group : MergeGroup := .operator 105870 105827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105870) (leftOrdinal := 1)
    (rightResult := 105827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46990⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105875

namespace LeftMerge105877
def owner : Owner := ⟨.program ⟨257⟩, ⟨46993⟩⟩
def mergeEvent : Nat := 105877
def frameStart : Nat := 105782
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46475⟩⟩] } }
def rhsRaw : List Term := Proof.Events413.exact105824RawTerms
def group : MergeGroup := .relation 105876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105876) (rhsResult := 105824)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46990⟩⟩) ⟨46475⟩ 105824) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46475⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105877

namespace LeftMerge105885
def owner : Owner := ⟨.program ⟨257⟩, ⟨45478⟩⟩
def mergeEvent : Nat := 105885
def frameStart : Nat := 105782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events413.exact105838RawTerms
def rightRaw : List Term := Proof.Events413.exact105881RawTerms
def group : MergeGroup := .operator 105838 105881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105838) (leftOrdinal := 0)
    (rightResult := 105881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45476⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105885

namespace LeftMerge105902
def owner : Owner := ⟨.program ⟨257⟩, ⟨45922⟩⟩
def mergeEvent : Nat := 105902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events413.exact105899RawTerms
def group : MergeGroup := .relation 105901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105901) (rhsResult := 105899)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩) (none) 105899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105902

namespace LeftMerge105903
def owner : Owner := ⟨.program ⟨257⟩, ⟨45922⟩⟩
def mergeEvent : Nat := 105903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩] } }
def rhsRaw : List Term := Proof.Events413.exact105899RawTerms
def group : MergeGroup := .relation 105901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105901) (rhsResult := 105899)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩) (none) 105899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105903

namespace LeftMerge105904
def owner : Owner := ⟨.program ⟨257⟩, ⟨45922⟩⟩
def mergeEvent : Nat := 105904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46475⟩⟩] } }
def rhsRaw : List Term := Proof.Events413.exact105899RawTerms
def group : MergeGroup := .relation 105901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105901) (rhsResult := 105899)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩) (none) 105899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46475⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105904

namespace LeftMerge105905
def owner : Owner := ⟨.program ⟨257⟩, ⟨45922⟩⟩
def mergeEvent : Nat := 105905
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events413.exact105899RawTerms
def group : MergeGroup := .relation 105901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105901) (rhsResult := 105899)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩) (none) 105899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105905

namespace LeftMerge105910
def owner : Owner := ⟨.program ⟨257⟩, ⟨46992⟩⟩
def mergeEvent : Nat := 105910
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46475⟩⟩] } }
def leftRaw : List Term := Proof.Events413.exact105906RawTerms
def rightRaw : List Term := Proof.Events412.exact105720RawTerms
def group : MergeGroup := .operator 105906 105720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105906) (leftOrdinal := 2)
    (rightResult := 105720) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46475⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46475⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105910

namespace LeftMerge105911
def owner : Owner := ⟨.program ⟨257⟩, ⟨46992⟩⟩
def mergeEvent : Nat := 105911
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩] } }
def leftRaw : List Term := Proof.Events413.exact105906RawTerms
def rightRaw : List Term := Proof.Events412.exact105720RawTerms
def group : MergeGroup := .operator 105906 105720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105906) (leftOrdinal := 1)
    (rightResult := 105720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105911

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
