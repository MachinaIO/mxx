import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge166270
def owner : Owner := ⟨.program ⟨257⟩, ⟨36044⟩⟩
def mergeEvent : Nat := 166270
def frameStart : Nat := 166210
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events649.exact166266RawTerms
def rightRaw : List Term := Proof.Events649.exact166264RawTerms
def group : MergeGroup := .operator 166266 166264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166266) (leftOrdinal := 0)
    (rightResult := 166264) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166270

namespace LeftMerge166293
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 166293
def frameStart : Nat := 166210
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events649.exact166289RawTerms
def rightRaw : List Term := Proof.Events649.exact166286RawTerms
def group : MergeGroup := .operator 166289 166286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166289) (leftOrdinal := 0)
    (rightResult := 166286) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166293

namespace LeftMerge166302
def owner : Owner := ⟨.program ⟨257⟩, ⟨36306⟩⟩
def mergeEvent : Nat := 166302
def frameStart : Nat := 166210
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩] } }
def leftRaw : List Term := Proof.Events649.exact166298RawTerms
def rightRaw : List Term := Proof.Events649.exact166255RawTerms
def group : MergeGroup := .operator 166298 166255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166298) (leftOrdinal := 0)
    (rightResult := 166255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36303⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166302

namespace LeftMerge166303
def owner : Owner := ⟨.program ⟨257⟩, ⟨36306⟩⟩
def mergeEvent : Nat := 166303
def frameStart : Nat := 166210
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩] } }
def leftRaw : List Term := Proof.Events649.exact166298RawTerms
def rightRaw : List Term := Proof.Events649.exact166255RawTerms
def group : MergeGroup := .operator 166298 166255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166298) (leftOrdinal := 1)
    (rightResult := 166255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36303⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166303

namespace LeftMerge166305
def owner : Owner := ⟨.program ⟨257⟩, ⟨36306⟩⟩
def mergeEvent : Nat := 166305
def frameStart : Nat := 166210
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35773⟩⟩] } }
def rhsRaw : List Term := Proof.Events649.exact166252RawTerms
def group : MergeGroup := .relation 166304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166304) (rhsResult := 166252)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36303⟩⟩) ⟨35773⟩ 166252) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35773⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], [⟨.program ⟨257⟩, ⟨35773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166305

namespace LeftMerge166313
def owner : Owner := ⟨.program ⟨257⟩, ⟨34782⟩⟩
def mergeEvent : Nat := 166313
def frameStart : Nat := 166210
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events649.exact166266RawTerms
def rightRaw : List Term := Proof.Events649.exact166309RawTerms
def group : MergeGroup := .operator 166266 166309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166266) (leftOrdinal := 0)
    (rightResult := 166309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34780⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166313

namespace LeftMerge166330
def owner : Owner := ⟨.program ⟨257⟩, ⟨35232⟩⟩
def mergeEvent : Nat := 166330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events649.exact166327RawTerms
def group : MergeGroup := .relation 166329
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166329) (rhsResult := 166327)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 166328 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35229⟩⟩]⟩) (none) 166327) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166330

namespace LeftMerge166331
def owner : Owner := ⟨.program ⟨257⟩, ⟨35232⟩⟩
def mergeEvent : Nat := 166331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩] } }
def rhsRaw : List Term := Proof.Events649.exact166327RawTerms
def group : MergeGroup := .relation 166329
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166329) (rhsResult := 166327)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 166328 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35229⟩⟩]⟩) (none) 166327) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166331

namespace LeftMerge166332
def owner : Owner := ⟨.program ⟨257⟩, ⟨35232⟩⟩
def mergeEvent : Nat := 166332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35773⟩⟩] } }
def rhsRaw : List Term := Proof.Events649.exact166327RawTerms
def group : MergeGroup := .relation 166329
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166329) (rhsResult := 166327)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 166328 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35229⟩⟩]⟩) (none) 166327) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35773⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], [⟨.program ⟨257⟩, ⟨35773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166332

namespace LeftMerge166333
def owner : Owner := ⟨.program ⟨257⟩, ⟨35232⟩⟩
def mergeEvent : Nat := 166333
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨34780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events649.exact166327RawTerms
def group : MergeGroup := .relation 166329
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166329) (rhsResult := 166327)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 166328 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35229⟩⟩]⟩) (none) 166327) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨34780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166333

namespace LeftMerge166338
def owner : Owner := ⟨.program ⟨257⟩, ⟨36305⟩⟩
def mergeEvent : Nat := 166338
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35773⟩⟩] } }
def leftRaw : List Term := Proof.Events649.exact166334RawTerms
def rightRaw : List Term := Proof.Events649.exact166148RawTerms
def group : MergeGroup := .operator 166334 166148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166334) (leftOrdinal := 2)
    (rightResult := 166148) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35773⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35773⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13641⟩⟩, ⟨.program ⟨257⟩, ⟨34530⟩⟩], [⟨.program ⟨257⟩, ⟨35773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166338

namespace LeftMerge166339
def owner : Owner := ⟨.program ⟨257⟩, ⟨36305⟩⟩
def mergeEvent : Nat := 166339
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩] } }
def leftRaw : List Term := Proof.Events649.exact166334RawTerms
def rightRaw : List Term := Proof.Events649.exact166148RawTerms
def group : MergeGroup := .operator 166334 166148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166334) (leftOrdinal := 1)
    (rightResult := 166148) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36303⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166339

namespace LeftMerge166347
def owner : Owner := ⟨.program ⟨257⟩, ⟨36731⟩⟩
def mergeEvent : Nat := 166347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36729⟩⟩] } }
def leftRaw : List Term := Proof.Events649.exact166341RawTerms
def rightRaw : List Term := Proof.Events648.exact166064RawTerms
def group : MergeGroup := .operator 166341 166064
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166341) (leftOrdinal := 0)
    (rightResult := 166064) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36729⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36729⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166347

namespace LeftMerge166348
def owner : Owner := ⟨.program ⟨257⟩, ⟨36731⟩⟩
def mergeEvent : Nat := 166348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨34780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36729⟩⟩] } }
def leftRaw : List Term := Proof.Events649.exact166341RawTerms
def rightRaw : List Term := Proof.Events648.exact166064RawTerms
def group : MergeGroup := .operator 166341 166064
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 166341) (leftOrdinal := 1)
    (rightResult := 166064) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨34780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36729⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨34780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36729⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166348

namespace LeftMerge166350
def owner : Owner := ⟨.program ⟨257⟩, ⟨36731⟩⟩
def mergeEvent : Nat := 166350
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨34780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35937⟩⟩] } }
def rhsRaw : List Term := Proof.Events648.exact166061RawTerms
def group : MergeGroup := .relation 166349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 166349) (rhsResult := 166061)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨34780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36729⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36729⟩⟩) ⟨35937⟩ 166061) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35937⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨34780⟩⟩], [⟨.program ⟨257⟩, ⟨35937⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge166350

namespace LeftMerge166364
def owner : Owner := ⟨.program ⟨257⟩, ⟨35579⟩⟩
def mergeEvent : Nat := 166364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35576⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events649.exact166358RawTerms
def group : MergeGroup := .operator 163745 166358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 166358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35576⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35576⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge166364

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
