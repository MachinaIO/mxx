import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge305252
def owner : Owner := ⟨.program ⟨257⟩, ⟨41736⟩⟩
def mergeEvent : Nat := 305252
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41170⟩⟩] } }
def leftRaw : List Term := Proof.Events1192.exact305247RawTerms
def rightRaw : List Term := Proof.Events1191.exact305093RawTerms
def group : MergeGroup := .operator 305247 305093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 305247) (leftOrdinal := 2)
    (rightResult := 305093) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41170⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41170⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40028⟩⟩], [⟨.program ⟨257⟩, ⟨41170⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge305252

namespace LeftMerge305260
def owner : Owner := ⟨.program ⟨257⟩, ⟨41737⟩⟩
def mergeEvent : Nat := 305260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }
def leftRaw : List Term := Proof.Events1192.exact305254RawTerms
def rightRaw : List Term := Proof.Events060.exact15602RawTerms
def group : MergeGroup := .operator 305254 15602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 305254) (leftOrdinal := 0)
    (rightResult := 15602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge305260

namespace LeftMerge305261
def owner : Owner := ⟨.program ⟨257⟩, ⟨41737⟩⟩
def mergeEvent : Nat := 305261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }
def leftRaw : List Term := Proof.Events1192.exact305254RawTerms
def rightRaw : List Term := Proof.Events060.exact15602RawTerms
def group : MergeGroup := .operator 305254 15602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 305254) (leftOrdinal := 1)
    (rightResult := 15602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge305261

namespace LeftMerge305263
def owner : Owner := ⟨.program ⟨257⟩, ⟨41737⟩⟩
def mergeEvent : Nat := 305263
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15595RawTerms
def group : MergeGroup := .relation 305262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 305262) (rhsResult := 15595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7159⟩⟩) ⟨7045⟩ 15595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge305263

namespace LeftMerge305277
def owner : Owner := ⟨.program ⟨257⟩, ⟨39055⟩⟩
def mergeEvent : Nat := 305277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297093RawTerms
def rightRaw : List Term := Proof.Events1192.exact305271RawTerms
def group : MergeGroup := .operator 297093 305271
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297093) (leftOrdinal := 0)
    (rightResult := 305271) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39053⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge305277

namespace LeftMerge305278
def owner : Owner := ⟨.program ⟨257⟩, ⟨39055⟩⟩
def mergeEvent : Nat := 305278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩] } }
def leftRaw : List Term := Proof.Events1160.exact297093RawTerms
def rightRaw : List Term := Proof.Events1192.exact305271RawTerms
def group : MergeGroup := .operator 297093 305271
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297093) (leftOrdinal := 1)
    (rightResult := 305271) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39053⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge305278

namespace LeftMerge305280
def owner : Owner := ⟨.program ⟨257⟩, ⟨39055⟩⟩
def mergeEvent : Nat := 305280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38490⟩⟩] } }
def rhsRaw : List Term := Proof.Events1192.exact305268RawTerms
def group : MergeGroup := .relation 305279
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 305279) (rhsResult := 305268)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39053⟩⟩) ⟨38490⟩ 305268) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38490⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨38490⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge305280

namespace LeftMerge305294
def owner : Owner := ⟨.program ⟨257⟩, ⟨37975⟩⟩
def mergeEvent : Nat := 305294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37972⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1192.exact305288RawTerms
def group : MergeGroup := .operator 295195 305288
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 305288) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37972⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37972⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge305294

namespace LeftMerge305391
def owner : Owner := ⟨.program ⟨257⟩, ⟨38748⟩⟩
def mergeEvent : Nat := 305391
def frameStart : Nat := 305337
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1192.exact305387RawTerms
def rightRaw : List Term := Proof.Events1192.exact305385RawTerms
def group : MergeGroup := .operator 305387 305385
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 305387) (leftOrdinal := 0)
    (rightResult := 305385) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge305391

namespace LeftMerge305403
def owner : Owner := ⟨.program ⟨257⟩, ⟨39054⟩⟩
def mergeEvent : Nat := 305403
def frameStart : Nat := 305337
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩] } }
def leftRaw : List Term := Proof.Events1192.exact305399RawTerms
def rightRaw : List Term := Proof.Events1192.exact305376RawTerms
def group : MergeGroup := .operator 305399 305376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 305399) (leftOrdinal := 0)
    (rightResult := 305376) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39053⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge305403

namespace LeftMerge305404
def owner : Owner := ⟨.program ⟨257⟩, ⟨39054⟩⟩
def mergeEvent : Nat := 305404
def frameStart : Nat := 305337
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩] } }
def leftRaw : List Term := Proof.Events1192.exact305399RawTerms
def rightRaw : List Term := Proof.Events1192.exact305376RawTerms
def group : MergeGroup := .operator 305399 305376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 305399) (leftOrdinal := 1)
    (rightResult := 305376) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39053⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge305404

namespace LeftMerge305406
def owner : Owner := ⟨.program ⟨257⟩, ⟨39054⟩⟩
def mergeEvent : Nat := 305406
def frameStart : Nat := 305337
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38490⟩⟩] } }
def rhsRaw : List Term := Proof.Events1192.exact305373RawTerms
def group : MergeGroup := .relation 305405
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 305405) (rhsResult := 305373)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39053⟩⟩) ⟨38490⟩ 305373) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38490⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨38490⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge305406

namespace LeftMerge305414
def owner : Owner := ⟨.program ⟨257⟩, ⟨37511⟩⟩
def mergeEvent : Nat := 305414
def frameStart : Nat := 305337
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37509⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1192.exact305387RawTerms
def rightRaw : List Term := Proof.Events1193.exact305410RawTerms
def group : MergeGroup := .operator 305387 305410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 305387) (leftOrdinal := 0)
    (rightResult := 305410) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37509⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37509⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge305414

namespace LeftMerge305431
def owner : Owner := ⟨.program ⟨257⟩, ⟨37975⟩⟩
def mergeEvent : Nat := 305431
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩] } }
def rhsRaw : List Term := Proof.Events1193.exact305428RawTerms
def group : MergeGroup := .relation 305430
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 305430) (rhsResult := 305428)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37972⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 305429 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37972⟩⟩]⟩) (none) 305428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge305431

namespace LeftMerge305432
def owner : Owner := ⟨.program ⟨257⟩, ⟨37975⟩⟩
def mergeEvent : Nat := 305432
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩] } }
def rhsRaw : List Term := Proof.Events1193.exact305428RawTerms
def group : MergeGroup := .relation 305430
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 305430) (rhsResult := 305428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37972⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 305429 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37972⟩⟩]⟩) (none) 305428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39053⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge305432

namespace LeftMerge305433
def owner : Owner := ⟨.program ⟨257⟩, ⟨37975⟩⟩
def mergeEvent : Nat := 305433
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38490⟩⟩] } }
def rhsRaw : List Term := Proof.Events1193.exact305428RawTerms
def group : MergeGroup := .relation 305430
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 305430) (rhsResult := 305428)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37972⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 305429 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37972⟩⟩]⟩) (none) 305428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38490⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37348⟩⟩], [⟨.program ⟨257⟩, ⟨38490⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge305433

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
