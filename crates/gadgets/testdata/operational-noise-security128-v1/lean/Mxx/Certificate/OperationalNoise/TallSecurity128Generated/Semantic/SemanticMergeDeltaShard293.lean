import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge51198
def owner : Owner := ⟨.program ⟨257⟩, ⟨64240⟩⟩
def mergeEvent : Nat := 51198
def frameStart : Nat := 51138
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events199.exact51194RawTerms
def rightRaw : List Term := Proof.Events199.exact51192RawTerms
def group : MergeGroup := .operator 51194 51192
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51194) (leftOrdinal := 0)
    (rightResult := 51192) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51198

namespace LeftMerge51221
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 51221
def frameStart : Nat := 51138
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51217RawTerms
def rightRaw : List Term := Proof.Events200.exact51214RawTerms
def group : MergeGroup := .operator 51217 51214
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51217) (leftOrdinal := 0)
    (rightResult := 51214) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51221

namespace LeftMerge51230
def owner : Owner := ⟨.program ⟨257⟩, ⟨64530⟩⟩
def mergeEvent : Nat := 51230
def frameStart : Nat := 51138
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51226RawTerms
def rightRaw : List Term := Proof.Events199.exact51183RawTerms
def group : MergeGroup := .operator 51226 51183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51226) (leftOrdinal := 0)
    (rightResult := 51183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64527⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51230

namespace LeftMerge51231
def owner : Owner := ⟨.program ⟨257⟩, ⟨64530⟩⟩
def mergeEvent : Nat := 51231
def frameStart : Nat := 51138
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51226RawTerms
def rightRaw : List Term := Proof.Events199.exact51183RawTerms
def group : MergeGroup := .operator 51226 51183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51226) (leftOrdinal := 1)
    (rightResult := 51183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64527⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51231

namespace LeftMerge51233
def owner : Owner := ⟨.program ⟨257⟩, ⟨64530⟩⟩
def mergeEvent : Nat := 51233
def frameStart : Nat := 51138
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63977⟩⟩] } }
def rhsRaw : List Term := Proof.Events199.exact51180RawTerms
def group : MergeGroup := .relation 51232
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51232) (rhsResult := 51180)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64527⟩⟩) ⟨63977⟩ 51180) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63977⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], [⟨.program ⟨257⟩, ⟨63977⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51233

namespace LeftMerge51241
def owner : Owner := ⟨.program ⟨257⟩, ⟨62874⟩⟩
def mergeEvent : Nat := 51241
def frameStart : Nat := 51138
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events199.exact51194RawTerms
def rightRaw : List Term := Proof.Events200.exact51237RawTerms
def group : MergeGroup := .operator 51194 51237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51194) (leftOrdinal := 0)
    (rightResult := 51237) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51241

namespace LeftMerge51258
def owner : Owner := ⟨.program ⟨257⟩, ⟨63452⟩⟩
def mergeEvent : Nat := 51258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events200.exact51255RawTerms
def group : MergeGroup := .relation 51257
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51257) (rhsResult := 51255)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51256 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63449⟩⟩]⟩) (none) 51255) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51258

namespace LeftMerge51259
def owner : Owner := ⟨.program ⟨257⟩, ⟨63452⟩⟩
def mergeEvent : Nat := 51259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩] } }
def rhsRaw : List Term := Proof.Events200.exact51255RawTerms
def group : MergeGroup := .relation 51257
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51257) (rhsResult := 51255)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51256 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63449⟩⟩]⟩) (none) 51255) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51259

namespace LeftMerge51260
def owner : Owner := ⟨.program ⟨257⟩, ⟨63452⟩⟩
def mergeEvent : Nat := 51260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63977⟩⟩] } }
def rhsRaw : List Term := Proof.Events200.exact51255RawTerms
def group : MergeGroup := .relation 51257
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51257) (rhsResult := 51255)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51256 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63449⟩⟩]⟩) (none) 51255) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63977⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], [⟨.program ⟨257⟩, ⟨63977⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51260

namespace LeftMerge51261
def owner : Owner := ⟨.program ⟨257⟩, ⟨63452⟩⟩
def mergeEvent : Nat := 51261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events200.exact51255RawTerms
def group : MergeGroup := .relation 51257
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51257) (rhsResult := 51255)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51256 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63449⟩⟩]⟩) (none) 51255) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51261

namespace LeftMerge51266
def owner : Owner := ⟨.program ⟨257⟩, ⟨64529⟩⟩
def mergeEvent : Nat := 51266
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63977⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51262RawTerms
def rightRaw : List Term := Proof.Events199.exact51076RawTerms
def group : MergeGroup := .operator 51262 51076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51262) (leftOrdinal := 2)
    (rightResult := 51076) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63977⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63977⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25586⟩⟩, ⟨.program ⟨257⟩, ⟨62681⟩⟩], [⟨.program ⟨257⟩, ⟨63977⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51266

namespace LeftMerge51267
def owner : Owner := ⟨.program ⟨257⟩, ⟨64529⟩⟩
def mergeEvent : Nat := 51267
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51262RawTerms
def rightRaw : List Term := Proof.Events199.exact51076RawTerms
def group : MergeGroup := .operator 51262 51076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51262) (leftOrdinal := 1)
    (rightResult := 51076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64527⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51267

namespace LeftMerge51275
def owner : Owner := ⟨.program ⟨257⟩, ⟨65122⟩⟩
def mergeEvent : Nat := 51275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51269RawTerms
def rightRaw : List Term := Proof.Events199.exact50992RawTerms
def group : MergeGroup := .operator 51269 50992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51269) (leftOrdinal := 0)
    (rightResult := 50992) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65120⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51275

namespace LeftMerge51276
def owner : Owner := ⟨.program ⟨257⟩, ⟨65122⟩⟩
def mergeEvent : Nat := 51276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51269RawTerms
def rightRaw : List Term := Proof.Events199.exact50992RawTerms
def group : MergeGroup := .operator 51269 50992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51269) (leftOrdinal := 1)
    (rightResult := 50992) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65120⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51276

namespace LeftMerge51278
def owner : Owner := ⟨.program ⟨257⟩, ⟨65122⟩⟩
def mergeEvent : Nat := 51278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64153⟩⟩] } }
def rhsRaw : List Term := Proof.Events199.exact50989RawTerms
def group : MergeGroup := .relation 51277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51277) (rhsResult := 50989)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨65120⟩⟩) ⟨64153⟩ 50989) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64153⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨64153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51278

namespace LeftMerge51292
def owner : Owner := ⟨.program ⟨257⟩, ⟨63839⟩⟩
def mergeEvent : Nat := 51292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events200.exact51286RawTerms
def group : MergeGroup := .operator 46745 51286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 51286) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63836⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51292

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
