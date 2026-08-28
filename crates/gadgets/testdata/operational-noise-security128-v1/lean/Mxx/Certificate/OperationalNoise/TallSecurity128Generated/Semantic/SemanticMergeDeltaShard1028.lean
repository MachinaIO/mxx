import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge168260
def owner : Owner := ⟨.program ⟨257⟩, ⟨63412⟩⟩
def mergeEvent : Nat := 168260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25538⟩⟩, ⟨.program ⟨257⟩, ⟨62573⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63953⟩⟩] } }
def rhsRaw : List Term := Proof.Events657.exact168255RawTerms
def group : MergeGroup := .relation 168257
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168257) (rhsResult := 168255)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 168256 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63409⟩⟩]⟩) (none) 168255) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25538⟩⟩, ⟨.program ⟨257⟩, ⟨62573⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63953⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25538⟩⟩, ⟨.program ⟨257⟩, ⟨62573⟩⟩], [⟨.program ⟨257⟩, ⟨63953⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168260

namespace LeftMerge168261
def owner : Owner := ⟨.program ⟨257⟩, ⟨63412⟩⟩
def mergeEvent : Nat := 168261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events657.exact168255RawTerms
def group : MergeGroup := .relation 168257
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168257) (rhsResult := 168255)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 168256 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63409⟩⟩]⟩) (none) 168255) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168261

namespace LeftMerge168266
def owner : Owner := ⟨.program ⟨257⟩, ⟨64485⟩⟩
def mergeEvent : Nat := 168266
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25538⟩⟩, ⟨.program ⟨257⟩, ⟨62573⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63953⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168262RawTerms
def rightRaw : List Term := Proof.Events656.exact168076RawTerms
def group : MergeGroup := .operator 168262 168076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168262) (leftOrdinal := 2)
    (rightResult := 168076) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25538⟩⟩, ⟨.program ⟨257⟩, ⟨62573⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63953⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25538⟩⟩, ⟨.program ⟨257⟩, ⟨62573⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63953⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25538⟩⟩, ⟨.program ⟨257⟩, ⟨62573⟩⟩], [⟨.program ⟨257⟩, ⟨63953⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168266

namespace LeftMerge168267
def owner : Owner := ⟨.program ⟨257⟩, ⟨64485⟩⟩
def mergeEvent : Nat := 168267
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64483⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168262RawTerms
def rightRaw : List Term := Proof.Events656.exact168076RawTerms
def group : MergeGroup := .operator 168262 168076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168262) (leftOrdinal := 1)
    (rightResult := 168076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64483⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64483⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64483⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168267

namespace LeftMerge168275
def owner : Owner := ⟨.program ⟨257⟩, ⟨64998⟩⟩
def mergeEvent : Nat := 168275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168269RawTerms
def rightRaw : List Term := Proof.Events656.exact167992RawTerms
def group : MergeGroup := .operator 168269 167992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168269) (leftOrdinal := 0)
    (rightResult := 167992) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64996⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168275

namespace LeftMerge168276
def owner : Owner := ⟨.program ⟨257⟩, ⟨64998⟩⟩
def mergeEvent : Nat := 168276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168269RawTerms
def rightRaw : List Term := Proof.Events656.exact167992RawTerms
def group : MergeGroup := .operator 168269 167992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168269) (leftOrdinal := 1)
    (rightResult := 167992) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64996⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168276

namespace LeftMerge168278
def owner : Owner := ⟨.program ⟨257⟩, ⟨64998⟩⟩
def mergeEvent : Nat := 168278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64117⟩⟩] } }
def rhsRaw : List Term := Proof.Events656.exact167989RawTerms
def group : MergeGroup := .relation 168277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168277) (rhsResult := 167989)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64996⟩⟩) ⟨64117⟩ 167989) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64117⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨64117⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168278

namespace LeftMerge168292
def owner : Owner := ⟨.program ⟨257⟩, ⟨63759⟩⟩
def mergeEvent : Nat := 168292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63756⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events657.exact168286RawTerms
def group : MergeGroup := .operator 163745 168286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 168286) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63756⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168292

namespace LeftMerge168413
def owner : Owner := ⟨.program ⟨257⟩, ⟨64304⟩⟩
def mergeEvent : Nat := 168413
def frameStart : Nat := 168347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168409RawTerms
def rightRaw : List Term := Proof.Events657.exact168407RawTerms
def group : MergeGroup := .operator 168409 168407
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168409) (leftOrdinal := 0)
    (rightResult := 168407) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168413

namespace LeftMerge168425
def owner : Owner := ⟨.program ⟨257⟩, ⟨64997⟩⟩
def mergeEvent : Nat := 168425
def frameStart : Nat := 168347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168421RawTerms
def rightRaw : List Term := Proof.Events657.exact168398RawTerms
def group : MergeGroup := .operator 168421 168398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168421) (leftOrdinal := 0)
    (rightResult := 168398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64996⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168425

namespace LeftMerge168426
def owner : Owner := ⟨.program ⟨257⟩, ⟨64997⟩⟩
def mergeEvent : Nat := 168426
def frameStart : Nat := 168347
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168421RawTerms
def rightRaw : List Term := Proof.Events657.exact168398RawTerms
def group : MergeGroup := .operator 168421 168398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168421) (leftOrdinal := 1)
    (rightResult := 168398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64996⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168426

namespace LeftMerge168428
def owner : Owner := ⟨.program ⟨257⟩, ⟨64997⟩⟩
def mergeEvent : Nat := 168428
def frameStart : Nat := 168347
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64117⟩⟩] } }
def rhsRaw : List Term := Proof.Events657.exact168395RawTerms
def group : MergeGroup := .relation 168427
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168427) (rhsResult := 168395)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64996⟩⟩) ⟨64117⟩ 168395) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64117⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨64117⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168428

namespace LeftMerge168436
def owner : Owner := ⟨.program ⟨257⟩, ⟨63159⟩⟩
def mergeEvent : Nat := 168436
def frameStart : Nat := 168347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168409RawTerms
def rightRaw : List Term := Proof.Events657.exact168432RawTerms
def group : MergeGroup := .operator 168409 168432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168409) (leftOrdinal := 0)
    (rightResult := 168432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63157⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168436

namespace LeftMerge168453
def owner : Owner := ⟨.program ⟨257⟩, ⟨63759⟩⟩
def mergeEvent : Nat := 168453
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }
def rhsRaw : List Term := Proof.Events658.exact168450RawTerms
def group : MergeGroup := .relation 168452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168452) (rhsResult := 168450)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 168451 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63756⟩⟩]⟩) (none) 168450) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168453

namespace LeftMerge168454
def owner : Owner := ⟨.program ⟨257⟩, ⟨63759⟩⟩
def mergeEvent : Nat := 168454
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩] } }
def rhsRaw : List Term := Proof.Events658.exact168450RawTerms
def group : MergeGroup := .relation 168452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168452) (rhsResult := 168450)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 168451 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63756⟩⟩]⟩) (none) 168450) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64996⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge168454

namespace LeftMerge168455
def owner : Owner := ⟨.program ⟨257⟩, ⟨63759⟩⟩
def mergeEvent : Nat := 168455
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64117⟩⟩] } }
def rhsRaw : List Term := Proof.Events658.exact168450RawTerms
def group : MergeGroup := .relation 168452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 168452) (rhsResult := 168450)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 168451 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63756⟩⟩]⟩) (none) 168450) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64117⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨64117⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge168455

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
