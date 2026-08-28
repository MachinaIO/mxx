import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge282176
def owner : Owner := ⟨.program ⟨257⟩, ⟨41554⟩⟩
def mergeEvent : Nat := 282176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41073⟩⟩] } }
def rhsRaw : List Term := Proof.Events1101.exact282101RawTerms
def group : MergeGroup := .relation 282175
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282175) (rhsResult := 282101)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41553⟩⟩) ⟨41073⟩ 282101) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41073⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], [⟨.program ⟨257⟩, ⟨41073⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282176

namespace LeftMerge282177
def owner : Owner := ⟨.program ⟨257⟩, ⟨41554⟩⟩
def mergeEvent : Nat := 282177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩] } }
def leftRaw : List Term := Proof.Events1102.exact282168RawTerms
def rightRaw : List Term := Proof.Events1101.exact282104RawTerms
def group : MergeGroup := .operator 282168 282104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282168) (leftOrdinal := 0)
    (rightResult := 282104) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41553⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282177

namespace LeftMerge282191
def owner : Owner := ⟨.program ⟨257⟩, ⟨40492⟩⟩
def mergeEvent : Nat := 282191
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1102.exact282185RawTerms
def group : MergeGroup := .operator 280745 282185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 282185) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40489⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282191

namespace LeftMerge282270
def owner : Owner := ⟨.program ⟨257⟩, ⟨39651⟩⟩
def mergeEvent : Nat := 282270
def frameStart : Nat := 282240
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1102.exact282266RawTerms
def rightRaw : List Term := Proof.Events1102.exact282263RawTerms
def group : MergeGroup := .operator 282266 282263
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282266) (leftOrdinal := 0)
    (rightResult := 282263) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14091⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282270

namespace LeftMerge282300
def owner : Owner := ⟨.program ⟨257⟩, ⟨41364⟩⟩
def mergeEvent : Nat := 282300
def frameStart : Nat := 282240
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1102.exact282296RawTerms
def rightRaw : List Term := Proof.Events1102.exact282294RawTerms
def group : MergeGroup := .operator 282296 282294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282296) (leftOrdinal := 0)
    (rightResult := 282294) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282300

namespace LeftMerge282321
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 282321
def frameStart : Nat := 282240
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events1102.exact282317RawTerms
def rightRaw : List Term := Proof.Events1102.exact282314RawTerms
def group : MergeGroup := .operator 282317 282314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282317) (leftOrdinal := 0)
    (rightResult := 282314) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282321

namespace LeftMerge282330
def owner : Owner := ⟨.program ⟨257⟩, ⟨41556⟩⟩
def mergeEvent : Nat := 282330
def frameStart : Nat := 282240
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩] } }
def leftRaw : List Term := Proof.Events1102.exact282326RawTerms
def rightRaw : List Term := Proof.Events1102.exact282285RawTerms
def group : MergeGroup := .operator 282326 282285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282326) (leftOrdinal := 0)
    (rightResult := 282285) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282330

namespace LeftMerge282331
def owner : Owner := ⟨.program ⟨257⟩, ⟨41556⟩⟩
def mergeEvent : Nat := 282331
def frameStart : Nat := 282240
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩] } }
def leftRaw : List Term := Proof.Events1102.exact282326RawTerms
def rightRaw : List Term := Proof.Events1102.exact282285RawTerms
def group : MergeGroup := .operator 282326 282285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282326) (leftOrdinal := 1)
    (rightResult := 282285) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41553⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282331

namespace LeftMerge282333
def owner : Owner := ⟨.program ⟨257⟩, ⟨41556⟩⟩
def mergeEvent : Nat := 282333
def frameStart : Nat := 282240
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41073⟩⟩] } }
def rhsRaw : List Term := Proof.Events1102.exact282282RawTerms
def group : MergeGroup := .relation 282332
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282332) (rhsResult := 282282)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41553⟩⟩) ⟨41073⟩ 282282) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41073⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], [⟨.program ⟨257⟩, ⟨41073⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282333

namespace LeftMerge282341
def owner : Owner := ⟨.program ⟨257⟩, ⟨40062⟩⟩
def mergeEvent : Nat := 282341
def frameStart : Nat := 282240
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40060⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1102.exact282296RawTerms
def rightRaw : List Term := Proof.Events1102.exact282337RawTerms
def group : MergeGroup := .operator 282296 282337
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282296) (leftOrdinal := 0)
    (rightResult := 282337) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40060⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282341

namespace LeftMerge282358
def owner : Owner := ⟨.program ⟨257⟩, ⟨40492⟩⟩
def mergeEvent : Nat := 282358
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events1102.exact282355RawTerms
def group : MergeGroup := .relation 282357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282357) (rhsResult := 282355)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 282356 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩]⟩) (none) 282355) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282358

namespace LeftMerge282359
def owner : Owner := ⟨.program ⟨257⟩, ⟨40492⟩⟩
def mergeEvent : Nat := 282359
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩] } }
def rhsRaw : List Term := Proof.Events1102.exact282355RawTerms
def group : MergeGroup := .relation 282357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282357) (rhsResult := 282355)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 282356 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩]⟩) (none) 282355) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282359

namespace LeftMerge282360
def owner : Owner := ⟨.program ⟨257⟩, ⟨40492⟩⟩
def mergeEvent : Nat := 282360
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41073⟩⟩] } }
def rhsRaw : List Term := Proof.Events1102.exact282355RawTerms
def group : MergeGroup := .relation 282357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282357) (rhsResult := 282355)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 282356 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩]⟩) (none) 282355) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41073⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], [⟨.program ⟨257⟩, ⟨41073⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282360

namespace LeftMerge282361
def owner : Owner := ⟨.program ⟨257⟩, ⟨40492⟩⟩
def mergeEvent : Nat := 282361
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1102.exact282355RawTerms
def group : MergeGroup := .relation 282357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 282357) (rhsResult := 282355)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 282356 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40489⟩⟩]⟩) (none) 282355) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40060⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282361

namespace LeftMerge282366
def owner : Owner := ⟨.program ⟨257⟩, ⟨41555⟩⟩
def mergeEvent : Nat := 282366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41073⟩⟩] } }
def leftRaw : List Term := Proof.Events1102.exact282362RawTerms
def rightRaw : List Term := Proof.Events1102.exact282178RawTerms
def group : MergeGroup := .operator 282362 282178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282362) (leftOrdinal := 2)
    (rightResult := 282178) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41073⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41073⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], [⟨.program ⟨257⟩, ⟨41073⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge282366

namespace LeftMerge282367
def owner : Owner := ⟨.program ⟨257⟩, ⟨41555⟩⟩
def mergeEvent : Nat := 282367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩] } }
def leftRaw : List Term := Proof.Events1102.exact282362RawTerms
def rightRaw : List Term := Proof.Events1102.exact282178RawTerms
def group : MergeGroup := .operator 282362 282178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282362) (leftOrdinal := 1)
    (rightResult := 282178) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge282367

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
