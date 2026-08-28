import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge251958
def owner : Owner := ⟨.program ⟨257⟩, ⟨45041⟩⟩
def mergeEvent : Nat := 251958
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact251954RawTerms
def rightRaw : List Term := Proof.Events984.exact251924RawTerms
def group : MergeGroup := .operator 251954 251924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251954) (leftOrdinal := 1)
    (rightResult := 251924) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251958

namespace LeftMerge251966
def owner : Owner := ⟨.program ⟨257⟩, ⟨46925⟩⟩
def mergeEvent : Nat := 251966
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact251960RawTerms
def rightRaw : List Term := Proof.Events983.exact251896RawTerms
def group : MergeGroup := .operator 251960 251896
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251960) (leftOrdinal := 1)
    (rightResult := 251896) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46924⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251966

namespace LeftMerge251968
def owner : Owner := ⟨.program ⟨257⟩, ⟨46925⟩⟩
def mergeEvent : Nat := 251968
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46439⟩⟩] } }
def rhsRaw : List Term := Proof.Events983.exact251893RawTerms
def group : MergeGroup := .relation 251967
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251967) (rhsResult := 251893)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46924⟩⟩) ⟨46439⟩ 251893) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46439⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨46439⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251968

namespace LeftMerge251969
def owner : Owner := ⟨.program ⟨257⟩, ⟨46925⟩⟩
def mergeEvent : Nat := 251969
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact251960RawTerms
def rightRaw : List Term := Proof.Events983.exact251896RawTerms
def group : MergeGroup := .operator 251960 251896
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251960) (leftOrdinal := 0)
    (rightResult := 251896) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46924⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251969

namespace LeftMerge251983
def owner : Owner := ⟨.program ⟨257⟩, ⟨45862⟩⟩
def mergeEvent : Nat := 251983
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events984.exact251977RawTerms
def group : MergeGroup := .operator 251495 251977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 251977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45859⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251983

namespace LeftMerge252062
def owner : Owner := ⟨.program ⟨257⟩, ⟨45035⟩⟩
def mergeEvent : Nat := 252062
def frameStart : Nat := 252032
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events984.exact252058RawTerms
def rightRaw : List Term := Proof.Events984.exact252055RawTerms
def group : MergeGroup := .operator 252058 252055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252058) (leftOrdinal := 0)
    (rightResult := 252055) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14706⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252062

namespace LeftMerge252092
def owner : Owner := ⟨.program ⟨257⟩, ⟨46728⟩⟩
def mergeEvent : Nat := 252092
def frameStart : Nat := 252032
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact252088RawTerms
def rightRaw : List Term := Proof.Events984.exact252086RawTerms
def group : MergeGroup := .operator 252088 252086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252088) (leftOrdinal := 0)
    (rightResult := 252086) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252092

namespace LeftMerge252115
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 252115
def frameStart : Nat := 252032
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact252111RawTerms
def rightRaw : List Term := Proof.Events984.exact252108RawTerms
def group : MergeGroup := .operator 252111 252108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252111) (leftOrdinal := 0)
    (rightResult := 252108) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252115

namespace LeftMerge252124
def owner : Owner := ⟨.program ⟨257⟩, ⟨46927⟩⟩
def mergeEvent : Nat := 252124
def frameStart : Nat := 252032
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact252120RawTerms
def rightRaw : List Term := Proof.Events984.exact252077RawTerms
def group : MergeGroup := .operator 252120 252077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252120) (leftOrdinal := 0)
    (rightResult := 252077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46924⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252124

namespace LeftMerge252125
def owner : Owner := ⟨.program ⟨257⟩, ⟨46927⟩⟩
def mergeEvent : Nat := 252125
def frameStart : Nat := 252032
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact252120RawTerms
def rightRaw : List Term := Proof.Events984.exact252077RawTerms
def group : MergeGroup := .operator 252120 252077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252120) (leftOrdinal := 1)
    (rightResult := 252077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46924⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252125

namespace LeftMerge252127
def owner : Owner := ⟨.program ⟨257⟩, ⟨46927⟩⟩
def mergeEvent : Nat := 252127
def frameStart : Nat := 252032
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46439⟩⟩] } }
def rhsRaw : List Term := Proof.Events984.exact252074RawTerms
def group : MergeGroup := .relation 252126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252126) (rhsResult := 252074)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46924⟩⟩) ⟨46439⟩ 252074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46439⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨46439⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252127

namespace LeftMerge252135
def owner : Owner := ⟨.program ⟨257⟩, ⟨45430⟩⟩
def mergeEvent : Nat := 252135
def frameStart : Nat := 252032
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact252088RawTerms
def rightRaw : List Term := Proof.Events984.exact252131RawTerms
def group : MergeGroup := .operator 252088 252131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252088) (leftOrdinal := 0)
    (rightResult := 252131) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252135

namespace LeftMerge252152
def owner : Owner := ⟨.program ⟨257⟩, ⟨45862⟩⟩
def mergeEvent : Nat := 252152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events984.exact252149RawTerms
def group : MergeGroup := .relation 252151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252151) (rhsResult := 252149)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 252150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩]⟩) (none) 252149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252152

namespace LeftMerge252153
def owner : Owner := ⟨.program ⟨257⟩, ⟨45862⟩⟩
def mergeEvent : Nat := 252153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩] } }
def rhsRaw : List Term := Proof.Events984.exact252149RawTerms
def group : MergeGroup := .relation 252151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252151) (rhsResult := 252149)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 252150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩]⟩) (none) 252149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252153

namespace LeftMerge252154
def owner : Owner := ⟨.program ⟨257⟩, ⟨45862⟩⟩
def mergeEvent : Nat := 252154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46439⟩⟩] } }
def rhsRaw : List Term := Proof.Events984.exact252149RawTerms
def group : MergeGroup := .relation 252151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252151) (rhsResult := 252149)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 252150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩]⟩) (none) 252149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46439⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨46439⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252154

namespace LeftMerge252155
def owner : Owner := ⟨.program ⟨257⟩, ⟨45862⟩⟩
def mergeEvent : Nat := 252155
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events984.exact252149RawTerms
def group : MergeGroup := .relation 252151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252151) (rhsResult := 252149)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 252150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45859⟩⟩]⟩) (none) 252149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252155

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
