import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge71177
def owner : Owner := ⟨.program ⟨214⟩, ⟨19383⟩⟩
def mergeEvent : Nat := 71177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events278.exact71171RawTerms
def group : MergeGroup := .operator 65387 71171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 71171) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19380⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71177

namespace LeftMerge71256
def owner : Owner := ⟨.program ⟨214⟩, ⟨13765⟩⟩
def mergeEvent : Nat := 71256
def frameStart : Nat := 71226
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events278.exact71252RawTerms
def rightRaw : List Term := Proof.Events278.exact71249RawTerms
def group : MergeGroup := .operator 71252 71249
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71252) (leftOrdinal := 0)
    (rightResult := 71249) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71256

namespace LeftMerge71286
def owner : Owner := ⟨.program ⟨214⟩, ⟨13878⟩⟩
def mergeEvent : Nat := 71286
def frameStart : Nat := 71226
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events278.exact71282RawTerms
def rightRaw : List Term := Proof.Events278.exact71280RawTerms
def group : MergeGroup := .operator 71282 71280
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71282) (leftOrdinal := 0)
    (rightResult := 71280) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71286

namespace LeftMerge71309
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def mergeEvent : Nat := 71309
def frameStart : Nat := 71226
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }
def leftRaw : List Term := Proof.Events278.exact71305RawTerms
def rightRaw : List Term := Proof.Events278.exact71302RawTerms
def group : MergeGroup := .operator 71305 71302
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71305) (leftOrdinal := 0)
    (rightResult := 71302) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7846⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71309

namespace LeftMerge71318
def owner : Owner := ⟨.program ⟨214⟩, ⟨25910⟩⟩
def mergeEvent : Nat := 71318
def frameStart : Nat := 71226
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩] } }
def leftRaw : List Term := Proof.Events278.exact71314RawTerms
def rightRaw : List Term := Proof.Events278.exact71271RawTerms
def group : MergeGroup := .operator 71314 71271
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71314) (leftOrdinal := 0)
    (rightResult := 71271) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25907⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71318

namespace LeftMerge71319
def owner : Owner := ⟨.program ⟨214⟩, ⟨25910⟩⟩
def mergeEvent : Nat := 71319
def frameStart : Nat := 71226
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩] } }
def leftRaw : List Term := Proof.Events278.exact71314RawTerms
def rightRaw : List Term := Proof.Events278.exact71271RawTerms
def group : MergeGroup := .operator 71314 71271
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71314) (leftOrdinal := 1)
    (rightResult := 71271) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25907⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71319

namespace LeftMerge71321
def owner : Owner := ⟨.program ⟨214⟩, ⟨25910⟩⟩
def mergeEvent : Nat := 71321
def frameStart : Nat := 71226
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23498⟩⟩] } }
def rhsRaw : List Term := Proof.Events278.exact71268RawTerms
def group : MergeGroup := .relation 71320
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71320) (rhsResult := 71268)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25907⟩⟩) ⟨23498⟩ 71268) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23498⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71321

namespace LeftMerge71329
def owner : Owner := ⟨.program ⟨214⟩, ⟨15700⟩⟩
def mergeEvent : Nat := 71329
def frameStart : Nat := 71226
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15698⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events278.exact71282RawTerms
def rightRaw : List Term := Proof.Events278.exact71325RawTerms
def group : MergeGroup := .operator 71282 71325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71282) (leftOrdinal := 0)
    (rightResult := 71325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15698⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71329

namespace LeftMerge71346
def owner : Owner := ⟨.program ⟨214⟩, ⟨19383⟩⟩
def mergeEvent : Nat := 71346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }
def rhsRaw : List Term := Proof.Events278.exact71343RawTerms
def group : MergeGroup := .relation 71345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71345) (rhsResult := 71343)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 71344 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩) (none) 71343) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71346

namespace LeftMerge71347
def owner : Owner := ⟨.program ⟨214⟩, ⟨19383⟩⟩
def mergeEvent : Nat := 71347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩] } }
def rhsRaw : List Term := Proof.Events278.exact71343RawTerms
def group : MergeGroup := .relation 71345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71345) (rhsResult := 71343)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 71344 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩) (none) 71343) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71347

namespace LeftMerge71348
def owner : Owner := ⟨.program ⟨214⟩, ⟨19383⟩⟩
def mergeEvent : Nat := 71348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23498⟩⟩] } }
def rhsRaw : List Term := Proof.Events278.exact71343RawTerms
def group : MergeGroup := .relation 71345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71345) (rhsResult := 71343)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 71344 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩) (none) 71343) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23498⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71348

namespace LeftMerge71349
def owner : Owner := ⟨.program ⟨214⟩, ⟨19383⟩⟩
def mergeEvent : Nat := 71349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events278.exact71343RawTerms
def group : MergeGroup := .relation 71345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71345) (rhsResult := 71343)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 71344 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩) (none) 71343) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15698⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71349

namespace LeftMerge71354
def owner : Owner := ⟨.program ⟨214⟩, ⟨25909⟩⟩
def mergeEvent : Nat := 71354
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23498⟩⟩] } }
def leftRaw : List Term := Proof.Events278.exact71350RawTerms
def rightRaw : List Term := Proof.Events277.exact71164RawTerms
def group : MergeGroup := .operator 71350 71164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71350) (leftOrdinal := 2)
    (rightResult := 71164) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23498⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23498⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71354

namespace LeftMerge71355
def owner : Owner := ⟨.program ⟨214⟩, ⟨25909⟩⟩
def mergeEvent : Nat := 71355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩] } }
def leftRaw : List Term := Proof.Events278.exact71350RawTerms
def rightRaw : List Term := Proof.Events277.exact71164RawTerms
def group : MergeGroup := .operator 71350 71164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71350) (leftOrdinal := 1)
    (rightResult := 71164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71355

namespace LeftMerge71363
def owner : Owner := ⟨.program ⟨214⟩, ⟨27421⟩⟩
def mergeEvent : Nat := 71363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩] } }
def leftRaw : List Term := Proof.Events278.exact71357RawTerms
def rightRaw : List Term := Proof.Events277.exact71080RawTerms
def group : MergeGroup := .operator 71357 71080
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71357) (leftOrdinal := 0)
    (rightResult := 71080) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27419⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71363

namespace LeftMerge71364
def owner : Owner := ⟨.program ⟨214⟩, ⟨27421⟩⟩
def mergeEvent : Nat := 71364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩] } }
def leftRaw : List Term := Proof.Events278.exact71357RawTerms
def rightRaw : List Term := Proof.Events277.exact71080RawTerms
def group : MergeGroup := .operator 71357 71080
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71357) (leftOrdinal := 1)
    (rightResult := 71080) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27419⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71364

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
