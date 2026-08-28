import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge295346
def owner : Owner := ⟨.program ⟨257⟩, ⟨48492⟩⟩
def mergeEvent : Nat := 295346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events1153.exact295343RawTerms
def group : MergeGroup := .relation 295345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295345) (rhsResult := 295343)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 295344 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48489⟩⟩]⟩) (none) 295343) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295346

namespace LeftMerge295347
def owner : Owner := ⟨.program ⟨257⟩, ⟨48492⟩⟩
def mergeEvent : Nat := 295347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49549⟩⟩] } }
def rhsRaw : List Term := Proof.Events1153.exact295343RawTerms
def group : MergeGroup := .relation 295345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295345) (rhsResult := 295343)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 295344 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48489⟩⟩]⟩) (none) 295343) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49549⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49549⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295347

namespace LeftMerge295348
def owner : Owner := ⟨.program ⟨257⟩, ⟨48492⟩⟩
def mergeEvent : Nat := 295348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49089⟩⟩] } }
def rhsRaw : List Term := Proof.Events1153.exact295343RawTerms
def group : MergeGroup := .relation 295345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295345) (rhsResult := 295343)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 295344 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48489⟩⟩]⟩) (none) 295343) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49089⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], [⟨.program ⟨257⟩, ⟨49089⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295348

namespace LeftMerge295349
def owner : Owner := ⟨.program ⟨257⟩, ⟨48492⟩⟩
def mergeEvent : Nat := 295349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1153.exact295343RawTerms
def group : MergeGroup := .relation 295345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295345) (rhsResult := 295343)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 295344 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48489⟩⟩]⟩) (none) 295343) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295349

namespace LeftMerge295354
def owner : Owner := ⟨.program ⟨257⟩, ⟨49551⟩⟩
def mergeEvent : Nat := 295354
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49089⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295350RawTerms
def rightRaw : List Term := Proof.Events1153.exact295177RawTerms
def group : MergeGroup := .operator 295350 295177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295350) (leftOrdinal := 2)
    (rightResult := 295177) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49089⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49089⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], [⟨.program ⟨257⟩, ⟨49089⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295354

namespace LeftMerge295355
def owner : Owner := ⟨.program ⟨257⟩, ⟨49551⟩⟩
def mergeEvent : Nat := 295355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49549⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295350RawTerms
def rightRaw : List Term := Proof.Events1153.exact295177RawTerms
def group : MergeGroup := .operator 295350 295177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295350) (leftOrdinal := 1)
    (rightResult := 295177) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49549⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49549⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49549⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295355

namespace LeftMerge295363
def owner : Owner := ⟨.program ⟨257⟩, ⟨49781⟩⟩
def mergeEvent : Nat := 295363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295357RawTerms
def rightRaw : List Term := Proof.Events1152.exact295093RawTerms
def group : MergeGroup := .operator 295357 295093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295357) (leftOrdinal := 0)
    (rightResult := 295093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49779⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295363

namespace LeftMerge295364
def owner : Owner := ⟨.program ⟨257⟩, ⟨49781⟩⟩
def mergeEvent : Nat := 295364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295357RawTerms
def rightRaw : List Term := Proof.Events1152.exact295093RawTerms
def group : MergeGroup := .operator 295357 295093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295357) (leftOrdinal := 1)
    (rightResult := 295093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49779⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295364

namespace LeftMerge295366
def owner : Owner := ⟨.program ⟨257⟩, ⟨49781⟩⟩
def mergeEvent : Nat := 295366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49211⟩⟩] } }
def rhsRaw : List Term := Proof.Events1152.exact295090RawTerms
def group : MergeGroup := .relation 295365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295365) (rhsResult := 295090)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49779⟩⟩) ⟨49211⟩ 295090) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49211⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49211⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295366

namespace LeftMerge295380
def owner : Owner := ⟨.program ⟨257⟩, ⟨48699⟩⟩
def mergeEvent : Nat := 295380
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48696⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1153.exact295374RawTerms
def group : MergeGroup := .operator 295195 295374
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 295374) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48696⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295380

namespace LeftMerge295477
def owner : Owner := ⟨.program ⟨257⟩, ⟨49468⟩⟩
def mergeEvent : Nat := 295477
def frameStart : Nat := 295423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1154.exact295473RawTerms
def rightRaw : List Term := Proof.Events1154.exact295471RawTerms
def group : MergeGroup := .operator 295473 295471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295473) (leftOrdinal := 0)
    (rightResult := 295471) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295477

namespace LeftMerge295489
def owner : Owner := ⟨.program ⟨257⟩, ⟨49780⟩⟩
def mergeEvent : Nat := 295489
def frameStart : Nat := 295423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩] } }
def leftRaw : List Term := Proof.Events1154.exact295485RawTerms
def rightRaw : List Term := Proof.Events1154.exact295462RawTerms
def group : MergeGroup := .operator 295485 295462
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295485) (leftOrdinal := 0)
    (rightResult := 295462) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49779⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295489

namespace LeftMerge295490
def owner : Owner := ⟨.program ⟨257⟩, ⟨49780⟩⟩
def mergeEvent : Nat := 295490
def frameStart : Nat := 295423
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩] } }
def leftRaw : List Term := Proof.Events1154.exact295485RawTerms
def rightRaw : List Term := Proof.Events1154.exact295462RawTerms
def group : MergeGroup := .operator 295485 295462
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295485) (leftOrdinal := 1)
    (rightResult := 295462) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49779⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295490

namespace LeftMerge295492
def owner : Owner := ⟨.program ⟨257⟩, ⟨49780⟩⟩
def mergeEvent : Nat := 295492
def frameStart : Nat := 295423
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49211⟩⟩] } }
def rhsRaw : List Term := Proof.Events1154.exact295459RawTerms
def group : MergeGroup := .relation 295491
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295491) (rhsResult := 295459)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49779⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49779⟩⟩) ⟨49211⟩ 295459) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49211⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49211⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295492

namespace LeftMerge295500
def owner : Owner := ⟨.program ⟨257⟩, ⟨48234⟩⟩
def mergeEvent : Nat := 295500
def frameStart : Nat := 295423
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1154.exact295473RawTerms
def rightRaw : List Term := Proof.Events1154.exact295496RawTerms
def group : MergeGroup := .operator 295473 295496
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295473) (leftOrdinal := 0)
    (rightResult := 295496) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295500

namespace LeftMerge295517
def owner : Owner := ⟨.program ⟨257⟩, ⟨48699⟩⟩
def mergeEvent : Nat := 295517
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }
def rhsRaw : List Term := Proof.Events1154.exact295514RawTerms
def group : MergeGroup := .relation 295516
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295516) (rhsResult := 295514)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 295515 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48696⟩⟩]⟩) (none) 295514) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295517

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
