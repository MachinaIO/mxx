import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge93372
def owner : Owner := ⟨.program ⟨257⟩, ⟨36755⟩⟩
def mergeEvent : Nat := 93372
def frameStart : Nat := 93294
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93368RawTerms
def rightRaw : List Term := Proof.Events364.exact93345RawTerms
def group : MergeGroup := .operator 93368 93345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93368) (leftOrdinal := 0)
    (rightResult := 93345) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36754⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93372

namespace LeftMerge93373
def owner : Owner := ⟨.program ⟨257⟩, ⟨36755⟩⟩
def mergeEvent : Nat := 93373
def frameStart : Nat := 93294
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93368RawTerms
def rightRaw : List Term := Proof.Events364.exact93345RawTerms
def group : MergeGroup := .operator 93368 93345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93368) (leftOrdinal := 1)
    (rightResult := 93345) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36754⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93373

namespace LeftMerge93375
def owner : Owner := ⟨.program ⟨257⟩, ⟨36755⟩⟩
def mergeEvent : Nat := 93375
def frameStart : Nat := 93294
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35946⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93342RawTerms
def group : MergeGroup := .relation 93374
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93374) (rhsResult := 93342)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36754⟩⟩) ⟨35946⟩ 93342) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35946⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35946⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93375

namespace LeftMerge93383
def owner : Owner := ⟨.program ⟨257⟩, ⟨35029⟩⟩
def mergeEvent : Nat := 93383
def frameStart : Nat := 93294
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93356RawTerms
def rightRaw : List Term := Proof.Events364.exact93379RawTerms
def group : MergeGroup := .operator 93356 93379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93356) (leftOrdinal := 0)
    (rightResult := 93379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35028⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨35028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93383

namespace LeftMerge93400
def owner : Owner := ⟨.program ⟨257⟩, ⟨35599⟩⟩
def mergeEvent : Nat := 93400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93397RawTerms
def group : MergeGroup := .relation 93399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93399) (rhsResult := 93397)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩]⟩) (none) 93397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93400

namespace LeftMerge93401
def owner : Owner := ⟨.program ⟨257⟩, ⟨35599⟩⟩
def mergeEvent : Nat := 93401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93397RawTerms
def group : MergeGroup := .relation 93399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93399) (rhsResult := 93397)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩]⟩) (none) 93397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93401

namespace LeftMerge93402
def owner : Owner := ⟨.program ⟨257⟩, ⟨35599⟩⟩
def mergeEvent : Nat := 93402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35946⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93397RawTerms
def group : MergeGroup := .relation 93399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93399) (rhsResult := 93397)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩]⟩) (none) 93397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35946⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35946⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93402

namespace LeftMerge93403
def owner : Owner := ⟨.program ⟨257⟩, ⟨35599⟩⟩
def mergeEvent : Nat := 93403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93397RawTerms
def group : MergeGroup := .relation 93399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93399) (rhsResult := 93397)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35596⟩⟩]⟩) (none) 93397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93403

namespace LeftMerge93408
def owner : Owner := ⟨.program ⟨257⟩, ⟨36757⟩⟩
def mergeEvent : Nat := 93408
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93404RawTerms
def rightRaw : List Term := Proof.Events364.exact93226RawTerms
def group : MergeGroup := .operator 93404 93226
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93404) (leftOrdinal := 0)
    (rightResult := 93226) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93408

namespace LeftMerge93409
def owner : Owner := ⟨.program ⟨257⟩, ⟨36757⟩⟩
def mergeEvent : Nat := 93409
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35946⟩⟩] } }
def leftRaw : List Term := Proof.Events364.exact93404RawTerms
def rightRaw : List Term := Proof.Events364.exact93226RawTerms
def group : MergeGroup := .operator 93404 93226
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93404) (leftOrdinal := 2)
    (rightResult := 93226) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35946⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35946⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35946⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93409

namespace LeftMerge93435
def owner : Owner := ⟨.program ⟨257⟩, ⟨28897⟩⟩
def mergeEvent : Nat := 93435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events015.exact3972RawTerms
def rightRaw : List Term := Proof.Events353.exact90528RawTerms
def group : MergeGroup := .operator 3972 90528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3972) (leftOrdinal := 0)
    (rightResult := 90528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28894⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93435

namespace LeftMerge93440
def owner : Owner := ⟨.program ⟨257⟩, ⟨9913⟩⟩
def mergeEvent : Nat := 93440
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90398RawTerms
def rightRaw : List Term := Proof.Events078.exact20086RawTerms
def group : MergeGroup := .operator 90398 20086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90398) (leftOrdinal := 0)
    (rightResult := 20086) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93440

namespace LeftMerge93457
def owner : Owner := ⟨.program ⟨257⟩, ⟨28900⟩⟩
def mergeEvent : Nat := 93457
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events365.exact93451RawTerms
def rightRaw : List Term := Proof.Events015.exact3975RawTerms
def group : MergeGroup := .operator 93451 3975
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93451) (leftOrdinal := 1)
    (rightResult := 3975) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13356⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93457

namespace LeftMerge93458
def owner : Owner := ⟨.program ⟨257⟩, ⟨28900⟩⟩
def mergeEvent : Nat := 93458
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events365.exact93451RawTerms
def rightRaw : List Term := Proof.Events015.exact3975RawTerms
def group : MergeGroup := .operator 93451 3975
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93451) (leftOrdinal := 0)
    (rightResult := 3975) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13356⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93458

namespace LeftMerge93463
def owner : Owner := ⟨.program ⟨257⟩, ⟨13357⟩⟩
def mergeEvent : Nat := 93463
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events015.exact3975RawTerms
def rightRaw : List Term := Proof.Events353.exact90528RawTerms
def group : MergeGroup := .operator 3975 90528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3975) (leftOrdinal := 0)
    (rightResult := 90528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13356⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93463

namespace LeftMerge93468
def owner : Owner := ⟨.program ⟨257⟩, ⟨9930⟩⟩
def mergeEvent : Nat := 93468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90398RawTerms
def rightRaw : List Term := Proof.Events078.exact20127RawTerms
def group : MergeGroup := .operator 90398 20127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90398) (leftOrdinal := 0)
    (rightResult := 20127) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93468

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
