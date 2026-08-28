import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge153929
def owner : Owner := ⟨.program ⟨257⟩, ⟨61427⟩⟩
def mergeEvent : Nat := 153929
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩] } }
def leftRaw : List Term := Proof.Events601.exact153923RawTerms
def rightRaw : List Term := Proof.Events601.exact153859RawTerms
def group : MergeGroup := .operator 153923 153859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153923) (leftOrdinal := 1)
    (rightResult := 153859) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61426⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge153929

namespace LeftMerge153931
def owner : Owner := ⟨.program ⟨257⟩, ⟨61427⟩⟩
def mergeEvent : Nat := 153931
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60931⟩⟩] } }
def rhsRaw : List Term := Proof.Events601.exact153856RawTerms
def group : MergeGroup := .relation 153930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 153930) (rhsResult := 153856)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61426⟩⟩) ⟨60931⟩ 153856) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60931⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨60931⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge153931

namespace LeftMerge153932
def owner : Owner := ⟨.program ⟨257⟩, ⟨61427⟩⟩
def mergeEvent : Nat := 153932
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩] } }
def leftRaw : List Term := Proof.Events601.exact153923RawTerms
def rightRaw : List Term := Proof.Events601.exact153859RawTerms
def group : MergeGroup := .operator 153923 153859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 153923) (leftOrdinal := 0)
    (rightResult := 153859) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61426⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153932

namespace LeftMerge153946
def owner : Owner := ⟨.program ⟨257⟩, ⟨60362⟩⟩
def mergeEvent : Nat := 153946
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events601.exact153940RawTerms
def group : MergeGroup := .operator 149120 153940
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 153940) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60359⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge153946

namespace LeftMerge154025
def owner : Owner := ⟨.program ⟨257⟩, ⟨59405⟩⟩
def mergeEvent : Nat := 154025
def frameStart : Nat := 153995
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events601.exact154021RawTerms
def rightRaw : List Term := Proof.Events601.exact154018RawTerms
def group : MergeGroup := .operator 154021 154018
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154021) (leftOrdinal := 0)
    (rightResult := 154018) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25214⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154025

namespace LeftMerge154055
def owner : Owner := ⟨.program ⟨257⟩, ⟨61216⟩⟩
def mergeEvent : Nat := 154055
def frameStart : Nat := 153995
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events601.exact154051RawTerms
def rightRaw : List Term := Proof.Events601.exact154049RawTerms
def group : MergeGroup := .operator 154051 154049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154051) (leftOrdinal := 0)
    (rightResult := 154049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154055

namespace LeftMerge154078
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 154078
def frameStart : Nat := 153995
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events601.exact154074RawTerms
def rightRaw : List Term := Proof.Events601.exact154071RawTerms
def group : MergeGroup := .operator 154074 154071
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154074) (leftOrdinal := 0)
    (rightResult := 154071) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154078

namespace LeftMerge154087
def owner : Owner := ⟨.program ⟨257⟩, ⟨61429⟩⟩
def mergeEvent : Nat := 154087
def frameStart : Nat := 153995
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩] } }
def leftRaw : List Term := Proof.Events601.exact154083RawTerms
def rightRaw : List Term := Proof.Events601.exact154040RawTerms
def group : MergeGroup := .operator 154083 154040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154083) (leftOrdinal := 0)
    (rightResult := 154040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61426⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154087

namespace LeftMerge154088
def owner : Owner := ⟨.program ⟨257⟩, ⟨61429⟩⟩
def mergeEvent : Nat := 154088
def frameStart : Nat := 153995
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩] } }
def leftRaw : List Term := Proof.Events601.exact154083RawTerms
def rightRaw : List Term := Proof.Events601.exact154040RawTerms
def group : MergeGroup := .operator 154083 154040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154083) (leftOrdinal := 1)
    (rightResult := 154040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61426⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154088

namespace LeftMerge154090
def owner : Owner := ⟨.program ⟨257⟩, ⟨61429⟩⟩
def mergeEvent : Nat := 154090
def frameStart : Nat := 153995
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60931⟩⟩] } }
def rhsRaw : List Term := Proof.Events601.exact154037RawTerms
def group : MergeGroup := .relation 154089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154089) (rhsResult := 154037)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61426⟩⟩) ⟨60931⟩ 154037) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60931⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨60931⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154090

namespace LeftMerge154098
def owner : Owner := ⟨.program ⟨257⟩, ⟨59806⟩⟩
def mergeEvent : Nat := 154098
def frameStart : Nat := 153995
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events601.exact154051RawTerms
def rightRaw : List Term := Proof.Events601.exact154094RawTerms
def group : MergeGroup := .operator 154051 154094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154051) (leftOrdinal := 0)
    (rightResult := 154094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154098

namespace LeftMerge154115
def owner : Owner := ⟨.program ⟨257⟩, ⟨60362⟩⟩
def mergeEvent : Nat := 154115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events602.exact154112RawTerms
def group : MergeGroup := .relation 154114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154114) (rhsResult := 154112)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 154113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩) (none) 154112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154115

namespace LeftMerge154116
def owner : Owner := ⟨.program ⟨257⟩, ⟨60362⟩⟩
def mergeEvent : Nat := 154116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩] } }
def rhsRaw : List Term := Proof.Events602.exact154112RawTerms
def group : MergeGroup := .relation 154114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154114) (rhsResult := 154112)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 154113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩) (none) 154112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61426⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154116

namespace LeftMerge154117
def owner : Owner := ⟨.program ⟨257⟩, ⟨60362⟩⟩
def mergeEvent : Nat := 154117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60931⟩⟩] } }
def rhsRaw : List Term := Proof.Events602.exact154112RawTerms
def group : MergeGroup := .relation 154114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154114) (rhsResult := 154112)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 154113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩) (none) 154112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60931⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨60931⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge154117

namespace LeftMerge154118
def owner : Owner := ⟨.program ⟨257⟩, ⟨60362⟩⟩
def mergeEvent : Nat := 154118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events602.exact154112RawTerms
def group : MergeGroup := .relation 154114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 154114) (rhsResult := 154112)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 154113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60359⟩⟩]⟩) (none) 154112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154118

namespace LeftMerge154123
def owner : Owner := ⟨.program ⟨257⟩, ⟨61428⟩⟩
def mergeEvent : Nat := 154123
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60931⟩⟩] } }
def leftRaw : List Term := Proof.Events602.exact154119RawTerms
def rightRaw : List Term := Proof.Events601.exact153933RawTerms
def group : MergeGroup := .operator 154119 153933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154119) (leftOrdinal := 2)
    (rightResult := 153933) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60931⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60931⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨25214⟩⟩, ⟨.program ⟨257⟩, ⟨59404⟩⟩], [⟨.program ⟨257⟩, ⟨60931⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge154123

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
