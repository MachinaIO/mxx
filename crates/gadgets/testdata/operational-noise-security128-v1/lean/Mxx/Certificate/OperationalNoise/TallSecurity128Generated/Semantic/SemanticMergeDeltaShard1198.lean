import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge195038
def owner : Owner := ⟨.program ⟨257⟩, ⟨38716⟩⟩
def mergeEvent : Nat := 195038
def frameStart : Nat := 194978
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events761.exact195034RawTerms
def rightRaw : List Term := Proof.Events761.exact195032RawTerms
def group : MergeGroup := .operator 195034 195032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195034) (leftOrdinal := 0)
    (rightResult := 195032) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195038

namespace LeftMerge195061
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 195061
def frameStart : Nat := 194978
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events761.exact195057RawTerms
def rightRaw : List Term := Proof.Events761.exact195054RawTerms
def group : MergeGroup := .operator 195057 195054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195057) (leftOrdinal := 0)
    (rightResult := 195054) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195061

namespace LeftMerge195070
def owner : Owner := ⟨.program ⟨257⟩, ⟨38964⟩⟩
def mergeEvent : Nat := 195070
def frameStart : Nat := 194978
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩] } }
def leftRaw : List Term := Proof.Events761.exact195066RawTerms
def rightRaw : List Term := Proof.Events761.exact195023RawTerms
def group : MergeGroup := .operator 195066 195023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195066) (leftOrdinal := 0)
    (rightResult := 195023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38961⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195070

namespace LeftMerge195071
def owner : Owner := ⟨.program ⟨257⟩, ⟨38964⟩⟩
def mergeEvent : Nat := 195071
def frameStart : Nat := 194978
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩] } }
def leftRaw : List Term := Proof.Events761.exact195066RawTerms
def rightRaw : List Term := Proof.Events761.exact195023RawTerms
def group : MergeGroup := .operator 195066 195023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195066) (leftOrdinal := 1)
    (rightResult := 195023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38961⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195071

namespace LeftMerge195073
def owner : Owner := ⟨.program ⟨257⟩, ⟨38964⟩⟩
def mergeEvent : Nat := 195073
def frameStart : Nat := 194978
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38441⟩⟩] } }
def rhsRaw : List Term := Proof.Events761.exact195020RawTerms
def group : MergeGroup := .relation 195072
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195072) (rhsResult := 195020)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38961⟩⟩) ⟨38441⟩ 195020) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38441⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], [⟨.program ⟨257⟩, ⟨38441⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195073

namespace LeftMerge195081
def owner : Owner := ⟨.program ⟨257⟩, ⟨37446⟩⟩
def mergeEvent : Nat := 195081
def frameStart : Nat := 194978
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events761.exact195034RawTerms
def rightRaw : List Term := Proof.Events762.exact195077RawTerms
def group : MergeGroup := .operator 195034 195077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195034) (leftOrdinal := 0)
    (rightResult := 195077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37444⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195081

namespace LeftMerge195098
def owner : Owner := ⟨.program ⟨257⟩, ⟨37892⟩⟩
def mergeEvent : Nat := 195098
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events762.exact195095RawTerms
def group : MergeGroup := .relation 195097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195097) (rhsResult := 195095)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195096 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37889⟩⟩]⟩) (none) 195095) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195098

namespace LeftMerge195099
def owner : Owner := ⟨.program ⟨257⟩, ⟨37892⟩⟩
def mergeEvent : Nat := 195099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩] } }
def rhsRaw : List Term := Proof.Events762.exact195095RawTerms
def group : MergeGroup := .relation 195097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195097) (rhsResult := 195095)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195096 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37889⟩⟩]⟩) (none) 195095) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195099

namespace LeftMerge195100
def owner : Owner := ⟨.program ⟨257⟩, ⟨37892⟩⟩
def mergeEvent : Nat := 195100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38441⟩⟩] } }
def rhsRaw : List Term := Proof.Events762.exact195095RawTerms
def group : MergeGroup := .relation 195097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195097) (rhsResult := 195095)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195096 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37889⟩⟩]⟩) (none) 195095) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38441⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], [⟨.program ⟨257⟩, ⟨38441⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195100

namespace LeftMerge195101
def owner : Owner := ⟨.program ⟨257⟩, ⟨37892⟩⟩
def mergeEvent : Nat := 195101
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events762.exact195095RawTerms
def group : MergeGroup := .relation 195097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195097) (rhsResult := 195095)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195096 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37889⟩⟩]⟩) (none) 195095) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195101

namespace LeftMerge195106
def owner : Owner := ⟨.program ⟨257⟩, ⟨38963⟩⟩
def mergeEvent : Nat := 195106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38441⟩⟩] } }
def leftRaw : List Term := Proof.Events762.exact195102RawTerms
def rightRaw : List Term := Proof.Events761.exact194916RawTerms
def group : MergeGroup := .operator 195102 194916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195102) (leftOrdinal := 2)
    (rightResult := 194916) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38441⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38441⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13911⟩⟩, ⟨.program ⟨257⟩, ⟨37162⟩⟩], [⟨.program ⟨257⟩, ⟨38441⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195106

namespace LeftMerge195107
def owner : Owner := ⟨.program ⟨257⟩, ⟨38963⟩⟩
def mergeEvent : Nat := 195107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩] } }
def leftRaw : List Term := Proof.Events762.exact195102RawTerms
def rightRaw : List Term := Proof.Events761.exact194916RawTerms
def group : MergeGroup := .operator 195102 194916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195102) (leftOrdinal := 1)
    (rightResult := 194916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38961⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195107

namespace LeftMerge195115
def owner : Owner := ⟨.program ⟨257⟩, ⟨39361⟩⟩
def mergeEvent : Nat := 195115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39359⟩⟩] } }
def leftRaw : List Term := Proof.Events762.exact195109RawTerms
def rightRaw : List Term := Proof.Events761.exact194832RawTerms
def group : MergeGroup := .operator 195109 194832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195109) (leftOrdinal := 0)
    (rightResult := 194832) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39359⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39359⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195115

namespace LeftMerge195116
def owner : Owner := ⟨.program ⟨257⟩, ⟨39361⟩⟩
def mergeEvent : Nat := 195116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39359⟩⟩] } }
def leftRaw : List Term := Proof.Events762.exact195109RawTerms
def rightRaw : List Term := Proof.Events761.exact194832RawTerms
def group : MergeGroup := .operator 195109 194832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195109) (leftOrdinal := 1)
    (rightResult := 194832) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39359⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39359⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195116

namespace LeftMerge195118
def owner : Owner := ⟨.program ⟨257⟩, ⟨39361⟩⟩
def mergeEvent : Nat := 195118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38599⟩⟩] } }
def rhsRaw : List Term := Proof.Events761.exact194829RawTerms
def group : MergeGroup := .relation 195117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195117) (rhsResult := 194829)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39359⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39359⟩⟩) ⟨38599⟩ 194829) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38599⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], [⟨.program ⟨257⟩, ⟨38599⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195118

namespace LeftMerge195132
def owner : Owner := ⟨.program ⟨257⟩, ⟨38219⟩⟩
def mergeEvent : Nat := 195132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38216⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events762.exact195126RawTerms
def group : MergeGroup := .operator 192995 195126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 195126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38216⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38216⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195132

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
