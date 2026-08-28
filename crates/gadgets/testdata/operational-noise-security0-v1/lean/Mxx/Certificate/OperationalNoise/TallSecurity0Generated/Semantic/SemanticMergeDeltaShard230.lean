import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge38242
def owner : Owner := ⟨.program ⟨214⟩, ⟨19971⟩⟩
def mergeEvent : Nat := 38242
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23252⟩⟩] } }
def rhsRaw : List Term := Proof.Events149.exact38237RawTerms
def group : MergeGroup := .relation 38239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38239) (rhsResult := 38237)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38238 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩) (none) 38237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23252⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], [⟨.program ⟨214⟩, ⟨23252⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38242

namespace LeftMerge38243
def owner : Owner := ⟨.program ⟨214⟩, ⟨19971⟩⟩
def mergeEvent : Nat := 38243
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events149.exact38237RawTerms
def group : MergeGroup := .relation 38239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38239) (rhsResult := 38237)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38238 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩) (none) 38237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38243

namespace LeftMerge38248
def owner : Owner := ⟨.program ⟨214⟩, ⟨25462⟩⟩
def mergeEvent : Nat := 38248
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23252⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38244RawTerms
def rightRaw : List Term := Proof.Events148.exact38058RawTerms
def group : MergeGroup := .operator 38244 38058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38244) (leftOrdinal := 2)
    (rightResult := 38058) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23252⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23252⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9935⟩⟩, ⟨.program ⟨214⟩, ⟨12582⟩⟩], [⟨.program ⟨214⟩, ⟨23252⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38248

namespace LeftMerge38249
def owner : Owner := ⟨.program ⟨214⟩, ⟨25462⟩⟩
def mergeEvent : Nat := 38249
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38244RawTerms
def rightRaw : List Term := Proof.Events148.exact38058RawTerms
def group : MergeGroup := .operator 38244 38058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38244) (leftOrdinal := 1)
    (rightResult := 38058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25460⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38249

namespace LeftMerge38257
def owner : Owner := ⟨.program ⟨214⟩, ⟨29196⟩⟩
def mergeEvent : Nat := 38257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38251RawTerms
def rightRaw : List Term := Proof.Events148.exact37974RawTerms
def group : MergeGroup := .operator 38251 37974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38251) (leftOrdinal := 0)
    (rightResult := 37974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29194⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38257

namespace LeftMerge38258
def owner : Owner := ⟨.program ⟨214⟩, ⟨29196⟩⟩
def mergeEvent : Nat := 38258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38251RawTerms
def rightRaw : List Term := Proof.Events148.exact37974RawTerms
def group : MergeGroup := .operator 38251 37974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38251) (leftOrdinal := 1)
    (rightResult := 37974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29194⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38258

namespace LeftMerge38260
def owner : Owner := ⟨.program ⟨214⟩, ⟨29196⟩⟩
def mergeEvent : Nat := 38260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24546⟩⟩] } }
def rhsRaw : List Term := Proof.Events148.exact37971RawTerms
def group : MergeGroup := .relation 38259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38259) (rhsResult := 37971)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29194⟩⟩) ⟨24546⟩ 37971) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24546⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨24546⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38260

namespace LeftMerge38274
def owner : Owner := ⟨.program ⟨214⟩, ⟨22275⟩⟩
def mergeEvent : Nat := 38274
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22272⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events149.exact38268RawTerms
def group : MergeGroup := .operator 36137 38268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 38268) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22272⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22272⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38274

namespace LeftMerge38395
def owner : Owner := ⟨.program ⟨214⟩, ⟨16599⟩⟩
def mergeEvent : Nat := 38395
def frameStart : Nat := 38329
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38391RawTerms
def rightRaw : List Term := Proof.Events149.exact38389RawTerms
def group : MergeGroup := .operator 38391 38389
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38391) (leftOrdinal := 0)
    (rightResult := 38389) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38395

namespace LeftMerge38407
def owner : Owner := ⟨.program ⟨214⟩, ⟨29195⟩⟩
def mergeEvent : Nat := 38407
def frameStart : Nat := 38329
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩] } }
def leftRaw : List Term := Proof.Events150.exact38403RawTerms
def rightRaw : List Term := Proof.Events149.exact38380RawTerms
def group : MergeGroup := .operator 38403 38380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38403) (leftOrdinal := 0)
    (rightResult := 38380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29194⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38407

namespace LeftMerge38408
def owner : Owner := ⟨.program ⟨214⟩, ⟨29195⟩⟩
def mergeEvent : Nat := 38408
def frameStart : Nat := 38329
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩] } }
def leftRaw : List Term := Proof.Events150.exact38403RawTerms
def rightRaw : List Term := Proof.Events149.exact38380RawTerms
def group : MergeGroup := .operator 38403 38380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38403) (leftOrdinal := 1)
    (rightResult := 38380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29194⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38408

namespace LeftMerge38410
def owner : Owner := ⟨.program ⟨214⟩, ⟨29195⟩⟩
def mergeEvent : Nat := 38410
def frameStart : Nat := 38329
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24546⟩⟩] } }
def rhsRaw : List Term := Proof.Events149.exact38377RawTerms
def group : MergeGroup := .relation 38409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38409) (rhsResult := 38377)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29194⟩⟩) ⟨24546⟩ 38377) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24546⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨24546⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38410

namespace LeftMerge38418
def owner : Owner := ⟨.program ⟨214⟩, ⟨18212⟩⟩
def mergeEvent : Nat := 38418
def frameStart : Nat := 38329
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events149.exact38391RawTerms
def rightRaw : List Term := Proof.Events150.exact38414RawTerms
def group : MergeGroup := .operator 38391 38414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38391) (leftOrdinal := 0)
    (rightResult := 38414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38418

namespace LeftMerge38435
def owner : Owner := ⟨.program ⟨214⟩, ⟨22275⟩⟩
def mergeEvent : Nat := 38435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }
def rhsRaw : List Term := Proof.Events150.exact38432RawTerms
def group : MergeGroup := .relation 38434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38434) (rhsResult := 38432)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22272⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38433 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22272⟩⟩]⟩) (none) 38432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38435

namespace LeftMerge38436
def owner : Owner := ⟨.program ⟨214⟩, ⟨22275⟩⟩
def mergeEvent : Nat := 38436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩] } }
def rhsRaw : List Term := Proof.Events150.exact38432RawTerms
def group : MergeGroup := .relation 38434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38434) (rhsResult := 38432)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22272⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38433 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22272⟩⟩]⟩) (none) 38432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29194⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38436

namespace LeftMerge38437
def owner : Owner := ⟨.program ⟨214⟩, ⟨22275⟩⟩
def mergeEvent : Nat := 38437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24546⟩⟩] } }
def rhsRaw : List Term := Proof.Events150.exact38432RawTerms
def group : MergeGroup := .relation 38434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38434) (rhsResult := 38432)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22272⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38433 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22272⟩⟩]⟩) (none) 38432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24546⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨24546⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38437

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
