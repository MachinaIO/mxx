import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge105234
def owner : Owner := ⟨.program ⟨214⟩, ⟨28476⟩⟩
def mergeEvent : Nat := 105234
def frameStart : Nat := 105168
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105230RawTerms
def rightRaw : List Term := Proof.Events410.exact105207RawTerms
def group : MergeGroup := .operator 105230 105207
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105230) (leftOrdinal := 0)
    (rightResult := 105207) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28475⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105234

namespace LeftMerge105235
def owner : Owner := ⟨.program ⟨214⟩, ⟨28476⟩⟩
def mergeEvent : Nat := 105235
def frameStart : Nat := 105168
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16252⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105230RawTerms
def rightRaw : List Term := Proof.Events410.exact105207RawTerms
def group : MergeGroup := .operator 105230 105207
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105230) (leftOrdinal := 1)
    (rightResult := 105207) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16252⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28475⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105235

namespace LeftMerge105237
def owner : Owner := ⟨.program ⟨214⟩, ⟨28476⟩⟩
def mergeEvent : Nat := 105237
def frameStart : Nat := 105168
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16252⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24341⟩⟩] } }
def rhsRaw : List Term := Proof.Events410.exact105204RawTerms
def group : MergeGroup := .relation 105236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105236) (rhsResult := 105204)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28475⟩⟩) ⟨24341⟩ 105204) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24341⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105237

namespace LeftMerge105245
def owner : Owner := ⟨.program ⟨214⟩, ⟨17598⟩⟩
def mergeEvent : Nat := 105245
def frameStart : Nat := 105168
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17596⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105218RawTerms
def rightRaw : List Term := Proof.Events411.exact105241RawTerms
def group : MergeGroup := .operator 105218 105241
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105218) (leftOrdinal := 0)
    (rightResult := 105241) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17596⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105245

namespace LeftMerge105262
def owner : Owner := ⟨.program ⟨214⟩, ⟨21752⟩⟩
def mergeEvent : Nat := 105262
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩] } }
def rhsRaw : List Term := Proof.Events411.exact105259RawTerms
def group : MergeGroup := .relation 105261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105261) (rhsResult := 105259)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105260 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩) (none) 105259) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105262

namespace LeftMerge105263
def owner : Owner := ⟨.program ⟨214⟩, ⟨21752⟩⟩
def mergeEvent : Nat := 105263
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩] } }
def rhsRaw : List Term := Proof.Events411.exact105259RawTerms
def group : MergeGroup := .relation 105261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105261) (rhsResult := 105259)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105260 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩) (none) 105259) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105263

namespace LeftMerge105264
def owner : Owner := ⟨.program ⟨214⟩, ⟨21752⟩⟩
def mergeEvent : Nat := 105264
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24341⟩⟩] } }
def rhsRaw : List Term := Proof.Events411.exact105259RawTerms
def group : MergeGroup := .relation 105261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105261) (rhsResult := 105259)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105260 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩) (none) 105259) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16252⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24341⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105264

namespace LeftMerge105265
def owner : Owner := ⟨.program ⟨214⟩, ⟨21752⟩⟩
def mergeEvent : Nat := 105265
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events411.exact105259RawTerms
def group : MergeGroup := .relation 105261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105261) (rhsResult := 105259)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 105260 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩) (none) 105259) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17596⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105265

namespace LeftMerge105270
def owner : Owner := ⟨.program ⟨214⟩, ⟨28478⟩⟩
def mergeEvent : Nat := 105270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105266RawTerms
def rightRaw : List Term := Proof.Events410.exact105112RawTerms
def group : MergeGroup := .operator 105266 105112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105266) (leftOrdinal := 0)
    (rightResult := 105112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105270

namespace LeftMerge105271
def owner : Owner := ⟨.program ⟨214⟩, ⟨28478⟩⟩
def mergeEvent : Nat := 105271
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24341⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105266RawTerms
def rightRaw : List Term := Proof.Events410.exact105112RawTerms
def group : MergeGroup := .operator 105266 105112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105266) (leftOrdinal := 2)
    (rightResult := 105112) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24341⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24341⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105271

namespace LeftMerge105279
def owner : Owner := ⟨.program ⟨214⟩, ⟨28479⟩⟩
def mergeEvent : Nat := 105279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105273RawTerms
def rightRaw : List Term := Proof.Events022.exact5659RawTerms
def group : MergeGroup := .operator 105273 5659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105273) (leftOrdinal := 0)
    (rightResult := 5659) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6677⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105279

namespace LeftMerge105280
def owner : Owner := ⟨.program ⟨214⟩, ⟨28479⟩⟩
def mergeEvent : Nat := 105280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105273RawTerms
def rightRaw : List Term := Proof.Events022.exact5659RawTerms
def group : MergeGroup := .operator 105273 5659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105273) (leftOrdinal := 1)
    (rightResult := 5659) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6677⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105280

namespace LeftMerge105282
def owner : Owner := ⟨.program ⟨214⟩, ⟨28479⟩⟩
def mergeEvent : Nat := 105282
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events022.exact5652RawTerms
def group : MergeGroup := .relation 105281
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105281) (rhsResult := 5652)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6677⟩⟩) ⟨6610⟩ 5652) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105282

namespace LeftMerge105296
def owner : Owner := ⟨.program ⟨214⟩, ⟨28260⟩⟩
def mergeEvent : Nat := 105296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩] } }
def leftRaw : List Term := Proof.Events383.exact98096RawTerms
def rightRaw : List Term := Proof.Events411.exact105290RawTerms
def group : MergeGroup := .operator 98096 105290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98096) (leftOrdinal := 0)
    (rightResult := 105290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28258⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge105296

namespace LeftMerge105297
def owner : Owner := ⟨.program ⟨214⟩, ⟨28260⟩⟩
def mergeEvent : Nat := 105297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩] } }
def leftRaw : List Term := Proof.Events383.exact98096RawTerms
def rightRaw : List Term := Proof.Events411.exact105290RawTerms
def group : MergeGroup := .operator 98096 105290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98096) (leftOrdinal := 1)
    (rightResult := 105290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28258⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105297

namespace LeftMerge105299
def owner : Owner := ⟨.program ⟨214⟩, ⟨28260⟩⟩
def mergeEvent : Nat := 105299
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24278⟩⟩] } }
def rhsRaw : List Term := Proof.Events411.exact105287RawTerms
def group : MergeGroup := .relation 105298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 105298) (rhsResult := 105287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28258⟩⟩) ⟨24278⟩ 105287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24278⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge105299

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
