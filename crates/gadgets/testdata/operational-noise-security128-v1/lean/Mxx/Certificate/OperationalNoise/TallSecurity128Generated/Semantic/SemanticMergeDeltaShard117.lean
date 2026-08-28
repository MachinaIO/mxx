import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge23172
def owner : Owner := ⟨.program ⟨257⟩, ⟨55404⟩⟩
def mergeEvent : Nat := 23172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54937⟩⟩] } }
def rhsRaw : List Term := Proof.Events090.exact23078RawTerms
def group : MergeGroup := .relation 23171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23171) (rhsResult := 23078)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55403⟩⟩) ⟨54937⟩ 23078) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54937⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], [⟨.program ⟨257⟩, ⟨54937⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23172

namespace LeftMerge23173
def owner : Owner := ⟨.program ⟨257⟩, ⟨55404⟩⟩
def mergeEvent : Nat := 23173
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23164RawTerms
def rightRaw : List Term := Proof.Events090.exact23081RawTerms
def group : MergeGroup := .operator 23164 23081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23164) (leftOrdinal := 0)
    (rightResult := 23081) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55403⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23173

namespace LeftMerge23187
def owner : Owner := ⟨.program ⟨257⟩, ⟨54345⟩⟩
def mergeEvent : Nat := 23187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events090.exact23181RawTerms
def group : MergeGroup := .operator 17169 23181
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 23181) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54342⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23187

namespace LeftMerge23266
def owner : Owner := ⟨.program ⟨257⟩, ⟨53292⟩⟩
def mergeEvent : Nat := 23266
def frameStart : Nat := 23236
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events090.exact23262RawTerms
def rightRaw : List Term := Proof.Events090.exact23259RawTerms
def group : MergeGroup := .operator 23262 23259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23262) (leftOrdinal := 0)
    (rightResult := 23259) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24666⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23266

namespace LeftMerge23296
def owner : Owner := ⟨.program ⟨257⟩, ⟨55232⟩⟩
def mergeEvent : Nat := 23296
def frameStart : Nat := 23236
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23292RawTerms
def rightRaw : List Term := Proof.Events090.exact23290RawTerms
def group : MergeGroup := .operator 23292 23290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23292) (leftOrdinal := 0)
    (rightResult := 23290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23296

namespace LeftMerge23319
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 23319
def frameStart : Nat := 23236
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events091.exact23315RawTerms
def rightRaw : List Term := Proof.Events091.exact23312RawTerms
def group : MergeGroup := .operator 23315 23312
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23315) (leftOrdinal := 0)
    (rightResult := 23312) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23319

namespace LeftMerge23328
def owner : Owner := ⟨.program ⟨257⟩, ⟨55406⟩⟩
def mergeEvent : Nat := 23328
def frameStart : Nat := 23236
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩] } }
def leftRaw : List Term := Proof.Events091.exact23324RawTerms
def rightRaw : List Term := Proof.Events090.exact23281RawTerms
def group : MergeGroup := .operator 23324 23281
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23324) (leftOrdinal := 1)
    (rightResult := 23281) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55403⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23328

namespace LeftMerge23330
def owner : Owner := ⟨.program ⟨257⟩, ⟨55406⟩⟩
def mergeEvent : Nat := 23330
def frameStart : Nat := 23236
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54937⟩⟩] } }
def rhsRaw : List Term := Proof.Events090.exact23278RawTerms
def group : MergeGroup := .relation 23329
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23329) (rhsResult := 23278)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55403⟩⟩) ⟨54937⟩ 23278) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54937⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], [⟨.program ⟨257⟩, ⟨54937⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23330

namespace LeftMerge23331
def owner : Owner := ⟨.program ⟨257⟩, ⟨55406⟩⟩
def mergeEvent : Nat := 23331
def frameStart : Nat := 23236
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩] } }
def leftRaw : List Term := Proof.Events091.exact23324RawTerms
def rightRaw : List Term := Proof.Events090.exact23281RawTerms
def group : MergeGroup := .operator 23324 23281
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23324) (leftOrdinal := 0)
    (rightResult := 23281) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55403⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23331

namespace LeftMerge23339
def owner : Owner := ⟨.program ⟨257⟩, ⟨53800⟩⟩
def mergeEvent : Nat := 23339
def frameStart : Nat := 23236
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events090.exact23292RawTerms
def rightRaw : List Term := Proof.Events091.exact23335RawTerms
def group : MergeGroup := .operator 23292 23335
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23292) (leftOrdinal := 0)
    (rightResult := 23335) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53798⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23339

namespace LeftMerge23356
def owner : Owner := ⟨.program ⟨257⟩, ⟨54345⟩⟩
def mergeEvent : Nat := 23356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54937⟩⟩] } }
def rhsRaw : List Term := Proof.Events091.exact23353RawTerms
def group : MergeGroup := .relation 23355
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23355) (rhsResult := 23353)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23354 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩) (none) 23353) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54937⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], [⟨.program ⟨257⟩, ⟨54937⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23356

namespace LeftMerge23357
def owner : Owner := ⟨.program ⟨257⟩, ⟨54345⟩⟩
def mergeEvent : Nat := 23357
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩] } }
def rhsRaw : List Term := Proof.Events091.exact23353RawTerms
def group : MergeGroup := .relation 23355
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23355) (rhsResult := 23353)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23354 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩) (none) 23353) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23357

namespace LeftMerge23358
def owner : Owner := ⟨.program ⟨257⟩, ⟨54345⟩⟩
def mergeEvent : Nat := 23358
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events091.exact23353RawTerms
def group : MergeGroup := .relation 23355
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23355) (rhsResult := 23353)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23354 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩) (none) 23353) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53798⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23358

namespace LeftMerge23359
def owner : Owner := ⟨.program ⟨257⟩, ⟨54345⟩⟩
def mergeEvent : Nat := 23359
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events091.exact23353RawTerms
def group : MergeGroup := .relation 23355
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23355) (rhsResult := 23353)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23354 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩) (none) 23353) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23359

namespace LeftMerge23364
def owner : Owner := ⟨.program ⟨257⟩, ⟨55405⟩⟩
def mergeEvent : Nat := 23364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54937⟩⟩] } }
def leftRaw : List Term := Proof.Events091.exact23360RawTerms
def rightRaw : List Term := Proof.Events090.exact23174RawTerms
def group : MergeGroup := .operator 23360 23174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23360) (leftOrdinal := 2)
    (rightResult := 23174) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54937⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54937⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24666⟩⟩, ⟨.program ⟨257⟩, ⟨53291⟩⟩], [⟨.program ⟨257⟩, ⟨54937⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23364

namespace LeftMerge23365
def owner : Owner := ⟨.program ⟨257⟩, ⟨55405⟩⟩
def mergeEvent : Nat := 23365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩] } }
def leftRaw : List Term := Proof.Events091.exact23360RawTerms
def rightRaw : List Term := Proof.Events090.exact23174RawTerms
def group : MergeGroup := .operator 23360 23174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23360) (leftOrdinal := 1)
    (rightResult := 23174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55403⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23365

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
