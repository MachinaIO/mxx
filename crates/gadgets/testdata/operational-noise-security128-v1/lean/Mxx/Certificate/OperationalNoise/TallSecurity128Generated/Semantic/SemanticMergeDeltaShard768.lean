import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge127248
def owner : Owner := ⟨.program ⟨257⟩, ⟨23398⟩⟩
def mergeEvent : Nat := 127248
def frameStart : Nat := 127155
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127243RawTerms
def rightRaw : List Term := Proof.Events496.exact127200RawTerms
def group : MergeGroup := .operator 127243 127200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127243) (leftOrdinal := 1)
    (rightResult := 127200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127248

namespace LeftMerge127250
def owner : Owner := ⟨.program ⟨257⟩, ⟨23398⟩⟩
def mergeEvent : Nat := 127250
def frameStart : Nat := 127155
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22905⟩⟩] } }
def rhsRaw : List Term := Proof.Events496.exact127197RawTerms
def group : MergeGroup := .relation 127249
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127249) (rhsResult := 127197)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23395⟩⟩) ⟨22905⟩ 127197) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22905⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨22905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127250

namespace LeftMerge127258
def owner : Owner := ⟨.program ⟨257⟩, ⟨21778⟩⟩
def mergeEvent : Nat := 127258
def frameStart : Nat := 127155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events496.exact127211RawTerms
def rightRaw : List Term := Proof.Events497.exact127254RawTerms
def group : MergeGroup := .operator 127211 127254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127211) (leftOrdinal := 0)
    (rightResult := 127254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127258

namespace LeftMerge127275
def owner : Owner := ⟨.program ⟨257⟩, ⟨22332⟩⟩
def mergeEvent : Nat := 127275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events497.exact127272RawTerms
def group : MergeGroup := .relation 127274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127274) (rhsResult := 127272)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩]⟩) (none) 127272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127275

namespace LeftMerge127276
def owner : Owner := ⟨.program ⟨257⟩, ⟨22332⟩⟩
def mergeEvent : Nat := 127276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩] } }
def rhsRaw : List Term := Proof.Events497.exact127272RawTerms
def group : MergeGroup := .relation 127274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127274) (rhsResult := 127272)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩]⟩) (none) 127272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127276

namespace LeftMerge127277
def owner : Owner := ⟨.program ⟨257⟩, ⟨22332⟩⟩
def mergeEvent : Nat := 127277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22905⟩⟩] } }
def rhsRaw : List Term := Proof.Events497.exact127272RawTerms
def group : MergeGroup := .relation 127274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127274) (rhsResult := 127272)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩]⟩) (none) 127272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22905⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨22905⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127277

namespace LeftMerge127278
def owner : Owner := ⟨.program ⟨257⟩, ⟨22332⟩⟩
def mergeEvent : Nat := 127278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events497.exact127272RawTerms
def group : MergeGroup := .relation 127274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127274) (rhsResult := 127272)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩]⟩) (none) 127272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127278

namespace LeftMerge127283
def owner : Owner := ⟨.program ⟨257⟩, ⟨23397⟩⟩
def mergeEvent : Nat := 127283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22905⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127279RawTerms
def rightRaw : List Term := Proof.Events496.exact127093RawTerms
def group : MergeGroup := .operator 127279 127093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127279) (leftOrdinal := 2)
    (rightResult := 127093) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22905⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22905⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨22905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127283

namespace LeftMerge127284
def owner : Owner := ⟨.program ⟨257⟩, ⟨23397⟩⟩
def mergeEvent : Nat := 127284
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127279RawTerms
def rightRaw : List Term := Proof.Events496.exact127093RawTerms
def group : MergeGroup := .operator 127279 127093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127279) (leftOrdinal := 1)
    (rightResult := 127093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127284

namespace LeftMerge127292
def owner : Owner := ⟨.program ⟨257⟩, ⟨23750⟩⟩
def mergeEvent : Nat := 127292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23748⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127286RawTerms
def rightRaw : List Term := Proof.Events496.exact127009RawTerms
def group : MergeGroup := .operator 127286 127009
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127286) (leftOrdinal := 0)
    (rightResult := 127009) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23748⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127292

namespace LeftMerge127293
def owner : Owner := ⟨.program ⟨257⟩, ⟨23750⟩⟩
def mergeEvent : Nat := 127293
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23748⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127286RawTerms
def rightRaw : List Term := Proof.Events496.exact127009RawTerms
def group : MergeGroup := .operator 127286 127009
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127286) (leftOrdinal := 1)
    (rightResult := 127009) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23748⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23748⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127293

namespace LeftMerge127295
def owner : Owner := ⟨.program ⟨257⟩, ⟨23750⟩⟩
def mergeEvent : Nat := 127295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23045⟩⟩] } }
def rhsRaw : List Term := Proof.Events496.exact127006RawTerms
def group : MergeGroup := .relation 127294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127294) (rhsResult := 127006)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23748⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23748⟩⟩) ⟨23045⟩ 127006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23045⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21776⟩⟩], [⟨.program ⟨257⟩, ⟨23045⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127295

namespace LeftMerge127309
def owner : Owner := ⟨.program ⟨257⟩, ⟨22599⟩⟩
def mergeEvent : Nat := 127309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22596⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events497.exact127303RawTerms
def group : MergeGroup := .operator 119870 127303
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 127303) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22596⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22596⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127309

namespace LeftMerge127430
def owner : Owner := ⟨.program ⟨257⟩, ⟨23272⟩⟩
def mergeEvent : Nat := 127430
def frameStart : Nat := 127364
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127426RawTerms
def rightRaw : List Term := Proof.Events497.exact127424RawTerms
def group : MergeGroup := .operator 127426 127424
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127426) (leftOrdinal := 0)
    (rightResult := 127424) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127430

namespace LeftMerge127442
def owner : Owner := ⟨.program ⟨257⟩, ⟨23749⟩⟩
def mergeEvent : Nat := 127442
def frameStart : Nat := 127364
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23748⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127438RawTerms
def rightRaw : List Term := Proof.Events497.exact127415RawTerms
def group : MergeGroup := .operator 127438 127415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127438) (leftOrdinal := 0)
    (rightResult := 127415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23748⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127442

namespace LeftMerge127443
def owner : Owner := ⟨.program ⟨257⟩, ⟨23749⟩⟩
def mergeEvent : Nat := 127443
def frameStart : Nat := 127364
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23748⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127438RawTerms
def rightRaw : List Term := Proof.Events497.exact127415RawTerms
def group : MergeGroup := .operator 127438 127415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127438) (leftOrdinal := 1)
    (rightResult := 127415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23748⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23748⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127443

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
