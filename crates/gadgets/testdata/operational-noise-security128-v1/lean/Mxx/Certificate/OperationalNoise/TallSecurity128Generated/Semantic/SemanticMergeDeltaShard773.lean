import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge128149
def owner : Owner := ⟨.program ⟨257⟩, ⟨15379⟩⟩
def mergeEvent : Nat := 128149
def frameStart : Nat := 128119
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events500.exact128145RawTerms
def rightRaw : List Term := Proof.Events500.exact128142RawTerms
def group : MergeGroup := .operator 128145 128142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128145) (leftOrdinal := 0)
    (rightResult := 128142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12321⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128149

namespace LeftMerge128179
def owner : Owner := ⟨.program ⟨257⟩, ⟨17112⟩⟩
def mergeEvent : Nat := 128179
def frameStart : Nat := 128119
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events500.exact128175RawTerms
def rightRaw : List Term := Proof.Events500.exact128173RawTerms
def group : MergeGroup := .operator 128175 128173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128175) (leftOrdinal := 0)
    (rightResult := 128173) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128179

namespace LeftMerge128202
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 128202
def frameStart : Nat := 128119
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events500.exact128198RawTerms
def rightRaw : List Term := Proof.Events500.exact128195RawTerms
def group : MergeGroup := .operator 128198 128195
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128198) (leftOrdinal := 0)
    (rightResult := 128195) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128202

namespace LeftMerge128211
def owner : Owner := ⟨.program ⟨257⟩, ⟨17318⟩⟩
def mergeEvent : Nat := 128211
def frameStart : Nat := 128119
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩] } }
def leftRaw : List Term := Proof.Events500.exact128207RawTerms
def rightRaw : List Term := Proof.Events500.exact128164RawTerms
def group : MergeGroup := .operator 128207 128164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128207) (leftOrdinal := 0)
    (rightResult := 128164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17315⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128211

namespace LeftMerge128212
def owner : Owner := ⟨.program ⟨257⟩, ⟨17318⟩⟩
def mergeEvent : Nat := 128212
def frameStart : Nat := 128119
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩] } }
def leftRaw : List Term := Proof.Events500.exact128207RawTerms
def rightRaw : List Term := Proof.Events500.exact128164RawTerms
def group : MergeGroup := .operator 128207 128164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128207) (leftOrdinal := 1)
    (rightResult := 128164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17315⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128212

namespace LeftMerge128214
def owner : Owner := ⟨.program ⟨257⟩, ⟨17318⟩⟩
def mergeEvent : Nat := 128214
def frameStart : Nat := 128119
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16825⟩⟩] } }
def rhsRaw : List Term := Proof.Events500.exact128161RawTerms
def group : MergeGroup := .relation 128213
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128213) (rhsResult := 128161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17315⟩⟩) ⟨16825⟩ 128161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16825⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], [⟨.program ⟨257⟩, ⟨16825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128214

namespace LeftMerge128222
def owner : Owner := ⟨.program ⟨257⟩, ⟨15758⟩⟩
def mergeEvent : Nat := 128222
def frameStart : Nat := 128119
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events500.exact128175RawTerms
def rightRaw : List Term := Proof.Events500.exact128218RawTerms
def group : MergeGroup := .operator 128175 128218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128175) (leftOrdinal := 0)
    (rightResult := 128218) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128222

namespace LeftMerge128239
def owner : Owner := ⟨.program ⟨257⟩, ⟨16252⟩⟩
def mergeEvent : Nat := 128239
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events500.exact128236RawTerms
def group : MergeGroup := .relation 128238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128238) (rhsResult := 128236)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16249⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 128237 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16249⟩⟩]⟩) (none) 128236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128239

namespace LeftMerge128240
def owner : Owner := ⟨.program ⟨257⟩, ⟨16252⟩⟩
def mergeEvent : Nat := 128240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩] } }
def rhsRaw : List Term := Proof.Events500.exact128236RawTerms
def group : MergeGroup := .relation 128238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128238) (rhsResult := 128236)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16249⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 128237 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16249⟩⟩]⟩) (none) 128236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128240

namespace LeftMerge128241
def owner : Owner := ⟨.program ⟨257⟩, ⟨16252⟩⟩
def mergeEvent : Nat := 128241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16825⟩⟩] } }
def rhsRaw : List Term := Proof.Events500.exact128236RawTerms
def group : MergeGroup := .relation 128238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128238) (rhsResult := 128236)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16249⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 128237 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16249⟩⟩]⟩) (none) 128236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16825⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], [⟨.program ⟨257⟩, ⟨16825⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128241

namespace LeftMerge128242
def owner : Owner := ⟨.program ⟨257⟩, ⟨16252⟩⟩
def mergeEvent : Nat := 128242
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events500.exact128236RawTerms
def group : MergeGroup := .relation 128238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128238) (rhsResult := 128236)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16249⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 128237 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16249⟩⟩]⟩) (none) 128236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128242

namespace LeftMerge128247
def owner : Owner := ⟨.program ⟨257⟩, ⟨17317⟩⟩
def mergeEvent : Nat := 128247
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16825⟩⟩] } }
def leftRaw : List Term := Proof.Events500.exact128243RawTerms
def rightRaw : List Term := Proof.Events500.exact128057RawTerms
def group : MergeGroup := .operator 128243 128057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128243) (leftOrdinal := 2)
    (rightResult := 128057) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16825⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16825⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], [⟨.program ⟨257⟩, ⟨16825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128247

namespace LeftMerge128248
def owner : Owner := ⟨.program ⟨257⟩, ⟨17317⟩⟩
def mergeEvent : Nat := 128248
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩] } }
def leftRaw : List Term := Proof.Events500.exact128243RawTerms
def rightRaw : List Term := Proof.Events500.exact128057RawTerms
def group : MergeGroup := .operator 128243 128057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128243) (leftOrdinal := 1)
    (rightResult := 128057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17315⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128248

namespace LeftMerge128256
def owner : Owner := ⟨.program ⟨257⟩, ⟨17651⟩⟩
def mergeEvent : Nat := 128256
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩] } }
def leftRaw : List Term := Proof.Events500.exact128250RawTerms
def rightRaw : List Term := Proof.Events499.exact127973RawTerms
def group : MergeGroup := .operator 128250 127973
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128250) (leftOrdinal := 0)
    (rightResult := 127973) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17649⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128256

namespace LeftMerge128257
def owner : Owner := ⟨.program ⟨257⟩, ⟨17651⟩⟩
def mergeEvent : Nat := 128257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩] } }
def leftRaw : List Term := Proof.Events500.exact128250RawTerms
def rightRaw : List Term := Proof.Events499.exact127973RawTerms
def group : MergeGroup := .operator 128250 127973
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128250) (leftOrdinal := 1)
    (rightResult := 127973) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17649⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128257

namespace LeftMerge128259
def owner : Owner := ⟨.program ⟨257⟩, ⟨17651⟩⟩
def mergeEvent : Nat := 128259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16965⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127970RawTerms
def group : MergeGroup := .relation 128258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128258) (rhsResult := 127970)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17649⟩⟩) ⟨16965⟩ 127970) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16965⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128259

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
