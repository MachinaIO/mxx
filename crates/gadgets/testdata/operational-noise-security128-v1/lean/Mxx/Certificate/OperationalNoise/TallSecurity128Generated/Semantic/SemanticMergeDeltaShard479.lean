import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge80806
def owner : Owner := ⟨.program ⟨257⟩, ⟨61526⟩⟩
def mergeEvent : Nat := 80806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60985⟩⟩] } }
def rhsRaw : List Term := Proof.Events315.exact80731RawTerms
def group : MergeGroup := .relation 80805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80805) (rhsResult := 80731)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61525⟩⟩) ⟨60985⟩ 80731) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60985⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80806

namespace LeftMerge80807
def owner : Owner := ⟨.program ⟨257⟩, ⟨61526⟩⟩
def mergeEvent : Nat := 80807
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩] } }
def leftRaw : List Term := Proof.Events315.exact80798RawTerms
def rightRaw : List Term := Proof.Events315.exact80734RawTerms
def group : MergeGroup := .operator 80798 80734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80798) (leftOrdinal := 0)
    (rightResult := 80734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61525⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80807

namespace LeftMerge80821
def owner : Owner := ⟨.program ⟨257⟩, ⟨60452⟩⟩
def mergeEvent : Nat := 80821
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events315.exact80815RawTerms
def group : MergeGroup := .operator 75995 80815
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 80815) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60449⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80821

namespace LeftMerge80900
def owner : Owner := ⟨.program ⟨257⟩, ⟨59648⟩⟩
def mergeEvent : Nat := 80900
def frameStart : Nat := 80870
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events316.exact80896RawTerms
def rightRaw : List Term := Proof.Events315.exact80893RawTerms
def group : MergeGroup := .operator 80896 80893
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80896) (leftOrdinal := 0)
    (rightResult := 80893) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25322⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80900

namespace LeftMerge80930
def owner : Owner := ⟨.program ⟨257⟩, ⟨61252⟩⟩
def mergeEvent : Nat := 80930
def frameStart : Nat := 80870
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact80926RawTerms
def rightRaw : List Term := Proof.Events316.exact80924RawTerms
def group : MergeGroup := .operator 80926 80924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80926) (leftOrdinal := 0)
    (rightResult := 80924) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80930

namespace LeftMerge80953
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 80953
def frameStart : Nat := 80870
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact80949RawTerms
def rightRaw : List Term := Proof.Events316.exact80946RawTerms
def group : MergeGroup := .operator 80949 80946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80949) (leftOrdinal := 0)
    (rightResult := 80946) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80953

namespace LeftMerge80962
def owner : Owner := ⟨.program ⟨257⟩, ⟨61528⟩⟩
def mergeEvent : Nat := 80962
def frameStart : Nat := 80870
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact80958RawTerms
def rightRaw : List Term := Proof.Events316.exact80915RawTerms
def group : MergeGroup := .operator 80958 80915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80958) (leftOrdinal := 0)
    (rightResult := 80915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61525⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80962

namespace LeftMerge80963
def owner : Owner := ⟨.program ⟨257⟩, ⟨61528⟩⟩
def mergeEvent : Nat := 80963
def frameStart : Nat := 80870
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact80958RawTerms
def rightRaw : List Term := Proof.Events316.exact80915RawTerms
def group : MergeGroup := .operator 80958 80915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80958) (leftOrdinal := 1)
    (rightResult := 80915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61525⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80963

namespace LeftMerge80965
def owner : Owner := ⟨.program ⟨257⟩, ⟨61528⟩⟩
def mergeEvent : Nat := 80965
def frameStart : Nat := 80870
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60985⟩⟩] } }
def rhsRaw : List Term := Proof.Events316.exact80912RawTerms
def group : MergeGroup := .relation 80964
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80964) (rhsResult := 80912)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61525⟩⟩) ⟨60985⟩ 80912) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60985⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80965

namespace LeftMerge80973
def owner : Owner := ⟨.program ⟨257⟩, ⟨59878⟩⟩
def mergeEvent : Nat := 80973
def frameStart : Nat := 80870
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact80926RawTerms
def rightRaw : List Term := Proof.Events316.exact80969RawTerms
def group : MergeGroup := .operator 80926 80969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80926) (leftOrdinal := 0)
    (rightResult := 80969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59876⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80973

namespace LeftMerge80990
def owner : Owner := ⟨.program ⟨257⟩, ⟨60452⟩⟩
def mergeEvent : Nat := 80990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events316.exact80987RawTerms
def group : MergeGroup := .relation 80989
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80989) (rhsResult := 80987)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80988 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩) (none) 80987) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80990

namespace LeftMerge80991
def owner : Owner := ⟨.program ⟨257⟩, ⟨60452⟩⟩
def mergeEvent : Nat := 80991
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩] } }
def rhsRaw : List Term := Proof.Events316.exact80987RawTerms
def group : MergeGroup := .relation 80989
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80989) (rhsResult := 80987)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80988 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩) (none) 80987) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80991

namespace LeftMerge80992
def owner : Owner := ⟨.program ⟨257⟩, ⟨60452⟩⟩
def mergeEvent : Nat := 80992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60985⟩⟩] } }
def rhsRaw : List Term := Proof.Events316.exact80987RawTerms
def group : MergeGroup := .relation 80989
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80989) (rhsResult := 80987)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80988 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩) (none) 80987) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60985⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80992

namespace LeftMerge80993
def owner : Owner := ⟨.program ⟨257⟩, ⟨60452⟩⟩
def mergeEvent : Nat := 80993
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events316.exact80987RawTerms
def group : MergeGroup := .relation 80989
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80989) (rhsResult := 80987)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80988 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩) (none) 80987) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80993

namespace LeftMerge80998
def owner : Owner := ⟨.program ⟨257⟩, ⟨61527⟩⟩
def mergeEvent : Nat := 80998
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60985⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact80994RawTerms
def rightRaw : List Term := Proof.Events315.exact80808RawTerms
def group : MergeGroup := .operator 80994 80808
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80994) (leftOrdinal := 2)
    (rightResult := 80808) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60985⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80998

namespace LeftMerge80999
def owner : Owner := ⟨.program ⟨257⟩, ⟨61527⟩⟩
def mergeEvent : Nat := 80999
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩] } }
def leftRaw : List Term := Proof.Events316.exact80994RawTerms
def rightRaw : List Term := Proof.Events315.exact80808RawTerms
def group : MergeGroup := .operator 80994 80808
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80994) (leftOrdinal := 1)
    (rightResult := 80808) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80999

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
