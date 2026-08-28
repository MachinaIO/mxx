import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge23911
def owner : Owner := ⟨.program ⟨214⟩, ⟨25389⟩⟩
def mergeEvent : Nat := 23911
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23905RawTerms
def rightRaw : List Term := Proof.Events093.exact23841RawTerms
def group : MergeGroup := .operator 23905 23841
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23905) (leftOrdinal := 1)
    (rightResult := 23841) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25388⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23911

namespace LeftMerge23913
def owner : Owner := ⟨.program ⟨214⟩, ⟨25389⟩⟩
def mergeEvent : Nat := 23913
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }
def rhsRaw : List Term := Proof.Events093.exact23838RawTerms
def group : MergeGroup := .relation 23912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23912) (rhsResult := 23838)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25388⟩⟩) ⟨23212⟩ 23838) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨23212⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23913

namespace LeftMerge23914
def owner : Owner := ⟨.program ⟨214⟩, ⟨25389⟩⟩
def mergeEvent : Nat := 23914
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23905RawTerms
def rightRaw : List Term := Proof.Events093.exact23841RawTerms
def group : MergeGroup := .operator 23905 23841
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23905) (leftOrdinal := 0)
    (rightResult := 23841) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25388⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23914

namespace LeftMerge23928
def owner : Owner := ⟨.program ⟨214⟩, ⟨19903⟩⟩
def mergeEvent : Nat := 23928
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events093.exact23922RawTerms
def group : MergeGroup := .operator 21512 23922
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 23922) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19900⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23928

namespace LeftMerge24007
def owner : Owner := ⟨.program ⟨214⟩, ⟨12395⟩⟩
def mergeEvent : Nat := 24007
def frameStart : Nat := 23977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events093.exact24003RawTerms
def rightRaw : List Term := Proof.Events093.exact24000RawTerms
def group : MergeGroup := .operator 24003 24000
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24003) (leftOrdinal := 0)
    (rightResult := 24000) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9835⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24007

namespace LeftMerge24037
def owner : Owner := ⟨.program ⟨214⟩, ⟨12480⟩⟩
def mergeEvent : Nat := 24037
def frameStart : Nat := 23977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact24033RawTerms
def rightRaw : List Term := Proof.Events093.exact24031RawTerms
def group : MergeGroup := .operator 24033 24031
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24033) (leftOrdinal := 0)
    (rightResult := 24031) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24037

namespace LeftMerge24060
def owner : Owner := ⟨.program ⟨214⟩, ⟨7869⟩⟩
def mergeEvent : Nat := 24060
def frameStart : Nat := 23977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact24056RawTerms
def rightRaw : List Term := Proof.Events093.exact24053RawTerms
def group : MergeGroup := .operator 24056 24053
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24056) (leftOrdinal := 0)
    (rightResult := 24053) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7867⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24060

namespace LeftMerge24069
def owner : Owner := ⟨.program ⟨214⟩, ⟨25391⟩⟩
def mergeEvent : Nat := 24069
def frameStart : Nat := 23977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24065RawTerms
def rightRaw : List Term := Proof.Events093.exact24022RawTerms
def group : MergeGroup := .operator 24065 24022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24065) (leftOrdinal := 0)
    (rightResult := 24022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25388⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24069

namespace LeftMerge24070
def owner : Owner := ⟨.program ⟨214⟩, ⟨25391⟩⟩
def mergeEvent : Nat := 24070
def frameStart : Nat := 23977
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24065RawTerms
def rightRaw : List Term := Proof.Events093.exact24022RawTerms
def group : MergeGroup := .operator 24065 24022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24065) (leftOrdinal := 1)
    (rightResult := 24022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25388⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24070

namespace LeftMerge24072
def owner : Owner := ⟨.program ⟨214⟩, ⟨25391⟩⟩
def mergeEvent : Nat := 24072
def frameStart : Nat := 23977
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }
def rhsRaw : List Term := Proof.Events093.exact24019RawTerms
def group : MergeGroup := .relation 24071
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24071) (rhsResult := 24019)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25388⟩⟩) ⟨23212⟩ 24019) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨23212⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24072

namespace LeftMerge24080
def owner : Owner := ⟨.program ⟨214⟩, ⟨16479⟩⟩
def mergeEvent : Nat := 24080
def frameStart : Nat := 23977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16477⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact24033RawTerms
def rightRaw : List Term := Proof.Events094.exact24076RawTerms
def group : MergeGroup := .operator 24033 24076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24033) (leftOrdinal := 0)
    (rightResult := 24076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16477⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16477⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24080

namespace LeftMerge24097
def owner : Owner := ⟨.program ⟨214⟩, ⟨19903⟩⟩
def mergeEvent : Nat := 24097
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }
def rhsRaw : List Term := Proof.Events094.exact24094RawTerms
def group : MergeGroup := .relation 24096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24096) (rhsResult := 24094)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24095 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩) (none) 24094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24097

namespace LeftMerge24098
def owner : Owner := ⟨.program ⟨214⟩, ⟨19903⟩⟩
def mergeEvent : Nat := 24098
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩] } }
def rhsRaw : List Term := Proof.Events094.exact24094RawTerms
def group : MergeGroup := .relation 24096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24096) (rhsResult := 24094)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24095 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩) (none) 24094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24098

namespace LeftMerge24099
def owner : Owner := ⟨.program ⟨214⟩, ⟨19903⟩⟩
def mergeEvent : Nat := 24099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }
def rhsRaw : List Term := Proof.Events094.exact24094RawTerms
def group : MergeGroup := .relation 24096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24096) (rhsResult := 24094)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24095 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩) (none) 24094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨23212⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24099

namespace LeftMerge24100
def owner : Owner := ⟨.program ⟨214⟩, ⟨19903⟩⟩
def mergeEvent : Nat := 24100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16477⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events094.exact24094RawTerms
def group : MergeGroup := .relation 24096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24096) (rhsResult := 24094)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24095 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩) (none) 24094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16477⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16477⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24100

namespace LeftMerge24105
def owner : Owner := ⟨.program ⟨214⟩, ⟨25390⟩⟩
def mergeEvent : Nat := 24105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }
def leftRaw : List Term := Proof.Events094.exact24101RawTerms
def rightRaw : List Term := Proof.Events093.exact23915RawTerms
def group : MergeGroup := .operator 24101 23915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24101) (leftOrdinal := 2)
    (rightResult := 23915) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨23212⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24105

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
