import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge196079
def owner : Owner := ⟨.program ⟨257⟩, ⟨31021⟩⟩
def mergeEvent : Nat := 196079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩] } }
def leftRaw : List Term := Proof.Events765.exact196073RawTerms
def rightRaw : List Term := Proof.Events764.exact195796RawTerms
def group : MergeGroup := .operator 196073 195796
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196073) (leftOrdinal := 0)
    (rightResult := 195796) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31019⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196079

namespace LeftMerge196080
def owner : Owner := ⟨.program ⟨257⟩, ⟨31021⟩⟩
def mergeEvent : Nat := 196080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩] } }
def leftRaw : List Term := Proof.Events765.exact196073RawTerms
def rightRaw : List Term := Proof.Events764.exact195796RawTerms
def group : MergeGroup := .operator 196073 195796
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196073) (leftOrdinal := 1)
    (rightResult := 195796) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31019⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196080

namespace LeftMerge196082
def owner : Owner := ⟨.program ⟨257⟩, ⟨31021⟩⟩
def mergeEvent : Nat := 196082
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30259⟩⟩] } }
def rhsRaw : List Term := Proof.Events764.exact195793RawTerms
def group : MergeGroup := .relation 196081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196081) (rhsResult := 195793)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31019⟩⟩) ⟨30259⟩ 195793) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30259⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨30259⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196082

namespace LeftMerge196096
def owner : Owner := ⟨.program ⟨257⟩, ⟨29879⟩⟩
def mergeEvent : Nat := 196096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events765.exact196090RawTerms
def group : MergeGroup := .operator 192995 196090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 196090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29876⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196096

namespace LeftMerge196217
def owner : Owner := ⟨.program ⟨257⟩, ⟨30456⟩⟩
def mergeEvent : Nat := 196217
def frameStart : Nat := 196151
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events766.exact196213RawTerms
def rightRaw : List Term := Proof.Events766.exact196211RawTerms
def group : MergeGroup := .operator 196213 196211
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196213) (leftOrdinal := 0)
    (rightResult := 196211) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196217

namespace LeftMerge196229
def owner : Owner := ⟨.program ⟨257⟩, ⟨31020⟩⟩
def mergeEvent : Nat := 196229
def frameStart : Nat := 196151
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩] } }
def leftRaw : List Term := Proof.Events766.exact196225RawTerms
def rightRaw : List Term := Proof.Events766.exact196202RawTerms
def group : MergeGroup := .operator 196225 196202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196225) (leftOrdinal := 0)
    (rightResult := 196202) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31019⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196229

namespace LeftMerge196230
def owner : Owner := ⟨.program ⟨257⟩, ⟨31020⟩⟩
def mergeEvent : Nat := 196230
def frameStart : Nat := 196151
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩] } }
def leftRaw : List Term := Proof.Events766.exact196225RawTerms
def rightRaw : List Term := Proof.Events766.exact196202RawTerms
def group : MergeGroup := .operator 196225 196202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196225) (leftOrdinal := 1)
    (rightResult := 196202) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31019⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196230

namespace LeftMerge196232
def owner : Owner := ⟨.program ⟨257⟩, ⟨31020⟩⟩
def mergeEvent : Nat := 196232
def frameStart : Nat := 196151
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30259⟩⟩] } }
def rhsRaw : List Term := Proof.Events766.exact196199RawTerms
def group : MergeGroup := .relation 196231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196231) (rhsResult := 196199)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31019⟩⟩) ⟨30259⟩ 196199) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30259⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨30259⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196232

namespace LeftMerge196240
def owner : Owner := ⟨.program ⟨257⟩, ⟨29326⟩⟩
def mergeEvent : Nat := 196240
def frameStart : Nat := 196151
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events766.exact196213RawTerms
def rightRaw : List Term := Proof.Events766.exact196236RawTerms
def group : MergeGroup := .operator 196213 196236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196213) (leftOrdinal := 0)
    (rightResult := 196236) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196240

namespace LeftMerge196257
def owner : Owner := ⟨.program ⟨257⟩, ⟨29879⟩⟩
def mergeEvent : Nat := 196257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }
def rhsRaw : List Term := Proof.Events766.exact196254RawTerms
def group : MergeGroup := .relation 196256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196256) (rhsResult := 196254)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 196255 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩]⟩) (none) 196254) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196257

namespace LeftMerge196258
def owner : Owner := ⟨.program ⟨257⟩, ⟨29879⟩⟩
def mergeEvent : Nat := 196258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩] } }
def rhsRaw : List Term := Proof.Events766.exact196254RawTerms
def group : MergeGroup := .relation 196256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196256) (rhsResult := 196254)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 196255 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩]⟩) (none) 196254) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196258

namespace LeftMerge196259
def owner : Owner := ⟨.program ⟨257⟩, ⟨29879⟩⟩
def mergeEvent : Nat := 196259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30259⟩⟩] } }
def rhsRaw : List Term := Proof.Events766.exact196254RawTerms
def group : MergeGroup := .relation 196256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196256) (rhsResult := 196254)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 196255 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩]⟩) (none) 196254) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30259⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨30259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196259

namespace LeftMerge196260
def owner : Owner := ⟨.program ⟨257⟩, ⟨29879⟩⟩
def mergeEvent : Nat := 196260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events766.exact196254RawTerms
def group : MergeGroup := .relation 196256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 196256) (rhsResult := 196254)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 196255 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29876⟩⟩]⟩) (none) 196254) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196260

namespace LeftMerge196265
def owner : Owner := ⟨.program ⟨257⟩, ⟨31022⟩⟩
def mergeEvent : Nat := 196265
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩] } }
def leftRaw : List Term := Proof.Events766.exact196261RawTerms
def rightRaw : List Term := Proof.Events765.exact196083RawTerms
def group : MergeGroup := .operator 196261 196083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196261) (leftOrdinal := 0)
    (rightResult := 196083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31019⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196265

namespace LeftMerge196266
def owner : Owner := ⟨.program ⟨257⟩, ⟨31022⟩⟩
def mergeEvent : Nat := 196266
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30259⟩⟩] } }
def leftRaw : List Term := Proof.Events766.exact196261RawTerms
def rightRaw : List Term := Proof.Events765.exact196083RawTerms
def group : MergeGroup := .operator 196261 196083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 196261) (leftOrdinal := 2)
    (rightResult := 196083) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30259⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29104⟩⟩], [⟨.program ⟨257⟩, ⟨30259⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge196266

namespace LeftMerge196292
def owner : Owner := ⟨.program ⟨257⟩, ⟨26145⟩⟩
def mergeEvent : Nat := 196292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events036.exact9231RawTerms
def rightRaw : List Term := Proof.Events753.exact192903RawTerms
def group : MergeGroup := .operator 9231 192903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9231) (leftOrdinal := 0)
    (rightResult := 192903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26142⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge196292

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
