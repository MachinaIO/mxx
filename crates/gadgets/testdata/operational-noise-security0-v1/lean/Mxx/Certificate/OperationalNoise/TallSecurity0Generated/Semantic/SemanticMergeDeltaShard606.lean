import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge98256
def owner : Owner := ⟨.program ⟨214⟩, ⟨21680⟩⟩
def mergeEvent : Nat := 98256
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }
def rhsRaw : List Term := Proof.Events383.exact98253RawTerms
def group : MergeGroup := .relation 98255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98255) (rhsResult := 98253)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98254 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩) (none) 98253) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98256

namespace LeftMerge98257
def owner : Owner := ⟨.program ⟨214⟩, ⟨21680⟩⟩
def mergeEvent : Nat := 98257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩] } }
def rhsRaw : List Term := Proof.Events383.exact98253RawTerms
def group : MergeGroup := .relation 98255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98255) (rhsResult := 98253)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98254 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩) (none) 98253) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98257

namespace LeftMerge98258
def owner : Owner := ⟨.program ⟨214⟩, ⟨21680⟩⟩
def mergeEvent : Nat := 98258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24279⟩⟩] } }
def rhsRaw : List Term := Proof.Events383.exact98253RawTerms
def group : MergeGroup := .relation 98255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98255) (rhsResult := 98253)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98254 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩) (none) 98253) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16168⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24279⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98258

namespace LeftMerge98259
def owner : Owner := ⟨.program ⟨214⟩, ⟨21680⟩⟩
def mergeEvent : Nat := 98259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events383.exact98253RawTerms
def group : MergeGroup := .relation 98255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98255) (rhsResult := 98253)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98254 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩) (none) 98253) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18303⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98259

namespace LeftMerge98264
def owner : Owner := ⟨.program ⟨214⟩, ⟨28268⟩⟩
def mergeEvent : Nat := 98264
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩] } }
def leftRaw : List Term := Proof.Events383.exact98260RawTerms
def rightRaw : List Term := Proof.Events383.exact98106RawTerms
def group : MergeGroup := .operator 98260 98106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98260) (leftOrdinal := 0)
    (rightResult := 98106) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98264

namespace LeftMerge98265
def owner : Owner := ⟨.program ⟨214⟩, ⟨28268⟩⟩
def mergeEvent : Nat := 98265
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24279⟩⟩] } }
def leftRaw : List Term := Proof.Events383.exact98260RawTerms
def rightRaw : List Term := Proof.Events383.exact98106RawTerms
def group : MergeGroup := .operator 98260 98106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98260) (leftOrdinal := 2)
    (rightResult := 98106) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24279⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24279⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98265

namespace LeftMerge98291
def owner : Owner := ⟨.program ⟨214⟩, ⟨11542⟩⟩
def mergeEvent : Nat := 98291
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events018.exact4773RawTerms
def rightRaw : List Term := Proof.Events000.exact32RawTerms
def group : MergeGroup := .operator 4773 32
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4773) (leftOrdinal := 0)
    (rightResult := 32) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11541⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98291

namespace LeftMerge98296
def owner : Owner := ⟨.program ⟨214⟩, ⟨7117⟩⟩
def mergeEvent : Nat := 98296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events042.exact10981RawTerms
def group : MergeGroup := .operator 27 10981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 10981) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98296

namespace LeftMerge98313
def owner : Owner := ⟨.program ⟨214⟩, ⟨14400⟩⟩
def mergeEvent : Nat := 98313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98307RawTerms
def rightRaw : List Term := Proof.Events018.exact4776RawTerms
def group : MergeGroup := .operator 98307 4776
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98307) (leftOrdinal := 1)
    (rightResult := 4776) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98313

namespace LeftMerge98314
def owner : Owner := ⟨.program ⟨214⟩, ⟨14400⟩⟩
def mergeEvent : Nat := 98314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98307RawTerms
def rightRaw : List Term := Proof.Events018.exact4776RawTerms
def group : MergeGroup := .operator 98307 4776
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98307) (leftOrdinal := 0)
    (rightResult := 4776) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98314

namespace LeftMerge98319
def owner : Owner := ⟨.program ⟨214⟩, ⟨14401⟩⟩
def mergeEvent : Nat := 98319
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events018.exact4776RawTerms
def rightRaw : List Term := Proof.Events000.exact32RawTerms
def group : MergeGroup := .operator 4776 32
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4776) (leftOrdinal := 0)
    (rightResult := 32) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98319

namespace LeftMerge98324
def owner : Owner := ⟨.program ⟨214⟩, ⟨7098⟩⟩
def mergeEvent : Nat := 98324
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events043.exact11022RawTerms
def group : MergeGroup := .operator 27 11022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 11022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98324

namespace LeftMerge98341
def owner : Owner := ⟨.program ⟨214⟩, ⟨14404⟩⟩
def mergeEvent : Nat := 98341
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98335RawTerms
def rightRaw : List Term := Proof.Events043.exact11011RawTerms
def group : MergeGroup := .operator 98335 11011
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98335) (leftOrdinal := 1)
    (rightResult := 11011) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7855⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98341

namespace LeftMerge98343
def owner : Owner := ⟨.program ⟨214⟩, ⟨14404⟩⟩
def mergeEvent : Nat := 98343
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def rhsRaw : List Term := Proof.Events042.exact10981RawTerms
def group : MergeGroup := .relation 98342
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98342) (rhsResult := 10981)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7855⟩⟩) ⟨6780⟩ 10981) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98343

namespace LeftMerge98344
def owner : Owner := ⟨.program ⟨214⟩, ⟨14404⟩⟩
def mergeEvent : Nat := 98344
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98335RawTerms
def rightRaw : List Term := Proof.Events043.exact11011RawTerms
def group : MergeGroup := .operator 98335 11011
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98335) (leftOrdinal := 0)
    (rightResult := 11011) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7855⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98344

namespace LeftMerge98349
def owner : Owner := ⟨.program ⟨214⟩, ⟨14405⟩⟩
def mergeEvent : Nat := 98349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98345RawTerms
def rightRaw : List Term := Proof.Events384.exact98315RawTerms
def group : MergeGroup := .operator 98345 98315
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98345) (leftOrdinal := 1)
    (rightResult := 98315) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98349

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
