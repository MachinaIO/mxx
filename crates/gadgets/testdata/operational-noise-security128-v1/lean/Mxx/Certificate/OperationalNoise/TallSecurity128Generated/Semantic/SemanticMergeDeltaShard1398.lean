import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge227248
def owner : Owner := ⟨.program ⟨257⟩, ⟨61450⟩⟩
def mergeEvent : Nat := 227248
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60943⟩⟩] } }
def leftRaw : List Term := Proof.Events887.exact227244RawTerms
def rightRaw : List Term := Proof.Events886.exact227058RawTerms
def group : MergeGroup := .operator 227244 227058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227244) (leftOrdinal := 2)
    (rightResult := 227058) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60943⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60943⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], [⟨.program ⟨257⟩, ⟨60943⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227248

namespace LeftMerge227249
def owner : Owner := ⟨.program ⟨257⟩, ⟨61450⟩⟩
def mergeEvent : Nat := 227249
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩] } }
def leftRaw : List Term := Proof.Events887.exact227244RawTerms
def rightRaw : List Term := Proof.Events886.exact227058RawTerms
def group : MergeGroup := .operator 227244 227058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227244) (leftOrdinal := 1)
    (rightResult := 227058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61448⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227249

namespace LeftMerge227257
def owner : Owner := ⟨.program ⟨257⟩, ⟨61863⟩⟩
def mergeEvent : Nat := 227257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩] } }
def leftRaw : List Term := Proof.Events887.exact227251RawTerms
def rightRaw : List Term := Proof.Events886.exact226974RawTerms
def group : MergeGroup := .operator 227251 226974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227251) (leftOrdinal := 0)
    (rightResult := 226974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61861⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227257

namespace LeftMerge227258
def owner : Owner := ⟨.program ⟨257⟩, ⟨61863⟩⟩
def mergeEvent : Nat := 227258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩] } }
def leftRaw : List Term := Proof.Events887.exact227251RawTerms
def rightRaw : List Term := Proof.Events886.exact226974RawTerms
def group : MergeGroup := .operator 227251 226974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227251) (leftOrdinal := 1)
    (rightResult := 226974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61861⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227258

namespace LeftMerge227260
def owner : Owner := ⟨.program ⟨257⟩, ⟨61863⟩⟩
def mergeEvent : Nat := 227260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61092⟩⟩] } }
def rhsRaw : List Term := Proof.Events886.exact226971RawTerms
def group : MergeGroup := .relation 227259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227259) (rhsResult := 226971)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61861⟩⟩) ⟨61092⟩ 226971) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61092⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227260

namespace LeftMerge227274
def owner : Owner := ⟨.program ⟨257⟩, ⟨60679⟩⟩
def mergeEvent : Nat := 227274
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events887.exact227268RawTerms
def group : MergeGroup := .operator 222245 227268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 227268) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60676⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227274

namespace LeftMerge227395
def owner : Owner := ⟨.program ⟨257⟩, ⟨61304⟩⟩
def mergeEvent : Nat := 227395
def frameStart : Nat := 227329
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227391RawTerms
def rightRaw : List Term := Proof.Events888.exact227389RawTerms
def group : MergeGroup := .operator 227391 227389
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227391) (leftOrdinal := 0)
    (rightResult := 227389) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227395

namespace LeftMerge227407
def owner : Owner := ⟨.program ⟨257⟩, ⟨61862⟩⟩
def mergeEvent : Nat := 227407
def frameStart : Nat := 227329
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227403RawTerms
def rightRaw : List Term := Proof.Events888.exact227380RawTerms
def group : MergeGroup := .operator 227403 227380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227403) (leftOrdinal := 0)
    (rightResult := 227380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61861⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227407

namespace LeftMerge227408
def owner : Owner := ⟨.program ⟨257⟩, ⟨61862⟩⟩
def mergeEvent : Nat := 227408
def frameStart : Nat := 227329
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227403RawTerms
def rightRaw : List Term := Proof.Events888.exact227380RawTerms
def group : MergeGroup := .operator 227403 227380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227403) (leftOrdinal := 1)
    (rightResult := 227380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61861⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227408

namespace LeftMerge227410
def owner : Owner := ⟨.program ⟨257⟩, ⟨61862⟩⟩
def mergeEvent : Nat := 227410
def frameStart : Nat := 227329
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61092⟩⟩] } }
def rhsRaw : List Term := Proof.Events888.exact227377RawTerms
def group : MergeGroup := .relation 227409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227409) (rhsResult := 227377)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61861⟩⟩) ⟨61092⟩ 227377) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61092⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227410

namespace LeftMerge227418
def owner : Owner := ⟨.program ⟨257⟩, ⟨60084⟩⟩
def mergeEvent : Nat := 227418
def frameStart : Nat := 227329
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227391RawTerms
def rightRaw : List Term := Proof.Events888.exact227414RawTerms
def group : MergeGroup := .operator 227391 227414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227391) (leftOrdinal := 0)
    (rightResult := 227414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227418

namespace LeftMerge227435
def owner : Owner := ⟨.program ⟨257⟩, ⟨60679⟩⟩
def mergeEvent : Nat := 227435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }
def rhsRaw : List Term := Proof.Events888.exact227432RawTerms
def group : MergeGroup := .relation 227434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227434) (rhsResult := 227432)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 227433 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩) (none) 227432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227435

namespace LeftMerge227436
def owner : Owner := ⟨.program ⟨257⟩, ⟨60679⟩⟩
def mergeEvent : Nat := 227436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩] } }
def rhsRaw : List Term := Proof.Events888.exact227432RawTerms
def group : MergeGroup := .relation 227434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227434) (rhsResult := 227432)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 227433 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩) (none) 227432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227436

namespace LeftMerge227437
def owner : Owner := ⟨.program ⟨257⟩, ⟨60679⟩⟩
def mergeEvent : Nat := 227437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61092⟩⟩] } }
def rhsRaw : List Term := Proof.Events888.exact227432RawTerms
def group : MergeGroup := .relation 227434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227434) (rhsResult := 227432)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 227433 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩) (none) 227432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61092⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227437

namespace LeftMerge227438
def owner : Owner := ⟨.program ⟨257⟩, ⟨60679⟩⟩
def mergeEvent : Nat := 227438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events888.exact227432RawTerms
def group : MergeGroup := .relation 227434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227434) (rhsResult := 227432)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 227433 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩) (none) 227432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227438

namespace LeftMerge227443
def owner : Owner := ⟨.program ⟨257⟩, ⟨61864⟩⟩
def mergeEvent : Nat := 227443
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227439RawTerms
def rightRaw : List Term := Proof.Events887.exact227261RawTerms
def group : MergeGroup := .operator 227439 227261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227439) (leftOrdinal := 0)
    (rightResult := 227261) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227443

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
