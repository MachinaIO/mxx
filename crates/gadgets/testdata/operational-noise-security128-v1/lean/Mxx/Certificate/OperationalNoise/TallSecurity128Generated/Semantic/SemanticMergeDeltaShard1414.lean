import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge230152
def owner : Owner := ⟨.program ⟨257⟩, ⟨20623⟩⟩
def mergeEvent : Nat := 230152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19852⟩⟩] } }
def rhsRaw : List Term := Proof.Events897.exact229863RawTerms
def group : MergeGroup := .relation 230151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230151) (rhsResult := 229863)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20621⟩⟩) ⟨19852⟩ 229863) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19852⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19852⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230152

namespace LeftMerge230166
def owner : Owner := ⟨.program ⟨257⟩, ⟨19439⟩⟩
def mergeEvent : Nat := 230166
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events899.exact230160RawTerms
def group : MergeGroup := .operator 222245 230160
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 230160) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19436⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230166

namespace LeftMerge230287
def owner : Owner := ⟨.program ⟨257⟩, ⟨20064⟩⟩
def mergeEvent : Nat := 230287
def frameStart : Nat := 230221
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events899.exact230283RawTerms
def rightRaw : List Term := Proof.Events899.exact230281RawTerms
def group : MergeGroup := .operator 230283 230281
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230283) (leftOrdinal := 0)
    (rightResult := 230281) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230287

namespace LeftMerge230299
def owner : Owner := ⟨.program ⟨257⟩, ⟨20622⟩⟩
def mergeEvent : Nat := 230299
def frameStart : Nat := 230221
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩] } }
def leftRaw : List Term := Proof.Events899.exact230295RawTerms
def rightRaw : List Term := Proof.Events899.exact230272RawTerms
def group : MergeGroup := .operator 230295 230272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230295) (leftOrdinal := 0)
    (rightResult := 230272) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20621⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230299

namespace LeftMerge230300
def owner : Owner := ⟨.program ⟨257⟩, ⟨20622⟩⟩
def mergeEvent : Nat := 230300
def frameStart : Nat := 230221
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩] } }
def leftRaw : List Term := Proof.Events899.exact230295RawTerms
def rightRaw : List Term := Proof.Events899.exact230272RawTerms
def group : MergeGroup := .operator 230295 230272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230295) (leftOrdinal := 1)
    (rightResult := 230272) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20621⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230300

namespace LeftMerge230302
def owner : Owner := ⟨.program ⟨257⟩, ⟨20622⟩⟩
def mergeEvent : Nat := 230302
def frameStart : Nat := 230221
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19852⟩⟩] } }
def rhsRaw : List Term := Proof.Events899.exact230269RawTerms
def group : MergeGroup := .relation 230301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230301) (rhsResult := 230269)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20621⟩⟩) ⟨19852⟩ 230269) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19852⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19852⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230302

namespace LeftMerge230310
def owner : Owner := ⟨.program ⟨257⟩, ⟨18849⟩⟩
def mergeEvent : Nat := 230310
def frameStart : Nat := 230221
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events899.exact230283RawTerms
def rightRaw : List Term := Proof.Events899.exact230306RawTerms
def group : MergeGroup := .operator 230283 230306
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230283) (leftOrdinal := 0)
    (rightResult := 230306) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230310

namespace LeftMerge230327
def owner : Owner := ⟨.program ⟨257⟩, ⟨19439⟩⟩
def mergeEvent : Nat := 230327
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }
def rhsRaw : List Term := Proof.Events899.exact230324RawTerms
def group : MergeGroup := .relation 230326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230326) (rhsResult := 230324)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230325 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩]⟩) (none) 230324) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230327

namespace LeftMerge230328
def owner : Owner := ⟨.program ⟨257⟩, ⟨19439⟩⟩
def mergeEvent : Nat := 230328
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩] } }
def rhsRaw : List Term := Proof.Events899.exact230324RawTerms
def group : MergeGroup := .relation 230326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230326) (rhsResult := 230324)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230325 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩]⟩) (none) 230324) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230328

namespace LeftMerge230329
def owner : Owner := ⟨.program ⟨257⟩, ⟨19439⟩⟩
def mergeEvent : Nat := 230329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19852⟩⟩] } }
def rhsRaw : List Term := Proof.Events899.exact230324RawTerms
def group : MergeGroup := .relation 230326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230326) (rhsResult := 230324)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230325 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩]⟩) (none) 230324) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19852⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19852⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230329

namespace LeftMerge230330
def owner : Owner := ⟨.program ⟨257⟩, ⟨19439⟩⟩
def mergeEvent : Nat := 230330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events899.exact230324RawTerms
def group : MergeGroup := .relation 230326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230326) (rhsResult := 230324)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230325 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19436⟩⟩]⟩) (none) 230324) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230330

namespace LeftMerge230335
def owner : Owner := ⟨.program ⟨257⟩, ⟨20624⟩⟩
def mergeEvent : Nat := 230335
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩] } }
def leftRaw : List Term := Proof.Events899.exact230331RawTerms
def rightRaw : List Term := Proof.Events899.exact230153RawTerms
def group : MergeGroup := .operator 230331 230153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230331) (leftOrdinal := 0)
    (rightResult := 230153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230335

namespace LeftMerge230336
def owner : Owner := ⟨.program ⟨257⟩, ⟨20624⟩⟩
def mergeEvent : Nat := 230336
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19852⟩⟩] } }
def leftRaw : List Term := Proof.Events899.exact230331RawTerms
def rightRaw : List Term := Proof.Events899.exact230153RawTerms
def group : MergeGroup := .operator 230331 230153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230331) (leftOrdinal := 2)
    (rightResult := 230153) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19852⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19852⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19852⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230336

namespace LeftMerge230362
def owner : Owner := ⟨.program ⟨257⟩, ⟨15453⟩⟩
def mergeEvent : Nat := 230362
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10957RawTerms
def rightRaw : List Term := Proof.Events867.exact222153RawTerms
def group : MergeGroup := .operator 10957 222153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10957) (leftOrdinal := 0)
    (rightResult := 222153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230362

namespace LeftMerge230367
def owner : Owner := ⟨.program ⟨257⟩, ⟨8496⟩⟩
def mergeEvent : Nat := 230367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222023RawTerms
def rightRaw : List Term := Proof.Events099.exact25597RawTerms
def group : MergeGroup := .operator 222023 25597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222023) (leftOrdinal := 0)
    (rightResult := 25597) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230367

namespace LeftMerge230384
def owner : Owner := ⟨.program ⟨257⟩, ⟨15456⟩⟩
def mergeEvent : Nat := 230384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events899.exact230378RawTerms
def rightRaw : List Term := Proof.Events042.exact10960RawTerms
def group : MergeGroup := .operator 230378 10960
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230378) (leftOrdinal := 1)
    (rightResult := 10960) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12366⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230384

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
