import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge58185
def owner : Owner := ⟨.program ⟨214⟩, ⟨26796⟩⟩
def mergeEvent : Nat := 58185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58178RawTerms
def rightRaw : List Term := Proof.Events226.exact57901RawTerms
def group : MergeGroup := .operator 58178 57901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58178) (leftOrdinal := 1)
    (rightResult := 57901) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26794⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58185

namespace LeftMerge58187
def owner : Owner := ⟨.program ⟨214⟩, ⟨26796⟩⟩
def mergeEvent : Nat := 58187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23850⟩⟩] } }
def rhsRaw : List Term := Proof.Events226.exact57898RawTerms
def group : MergeGroup := .relation 58186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58186) (rhsResult := 57898)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26794⟩⟩) ⟨23850⟩ 57898) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23850⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨23850⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58187

namespace LeftMerge58201
def owner : Owner := ⟨.program ⟨214⟩, ⟨20687⟩⟩
def mergeEvent : Nat := 58201
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events227.exact58195RawTerms
def group : MergeGroup := .operator 50762 58195
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 58195) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58201

namespace LeftMerge58322
def owner : Owner := ⟨.program ⟨214⟩, ⟨15160⟩⟩
def mergeEvent : Nat := 58322
def frameStart : Nat := 58256
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58318RawTerms
def rightRaw : List Term := Proof.Events227.exact58316RawTerms
def group : MergeGroup := .operator 58318 58316
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58318) (leftOrdinal := 0)
    (rightResult := 58316) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58322

namespace LeftMerge58334
def owner : Owner := ⟨.program ⟨214⟩, ⟨26795⟩⟩
def mergeEvent : Nat := 58334
def frameStart : Nat := 58256
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58330RawTerms
def rightRaw : List Term := Proof.Events227.exact58307RawTerms
def group : MergeGroup := .operator 58330 58307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58330) (leftOrdinal := 0)
    (rightResult := 58307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26794⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58334

namespace LeftMerge58335
def owner : Owner := ⟨.program ⟨214⟩, ⟨26795⟩⟩
def mergeEvent : Nat := 58335
def frameStart : Nat := 58256
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58330RawTerms
def rightRaw : List Term := Proof.Events227.exact58307RawTerms
def group : MergeGroup := .operator 58330 58307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58330) (leftOrdinal := 1)
    (rightResult := 58307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26794⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58335

namespace LeftMerge58337
def owner : Owner := ⟨.program ⟨214⟩, ⟨26795⟩⟩
def mergeEvent : Nat := 58337
def frameStart : Nat := 58256
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23850⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58304RawTerms
def group : MergeGroup := .relation 58336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58336) (rhsResult := 58304)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26794⟩⟩) ⟨23850⟩ 58304) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23850⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨23850⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58337

namespace LeftMerge58345
def owner : Owner := ⟨.program ⟨214⟩, ⟨15372⟩⟩
def mergeEvent : Nat := 58345
def frameStart : Nat := 58256
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15370⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58318RawTerms
def rightRaw : List Term := Proof.Events227.exact58341RawTerms
def group : MergeGroup := .operator 58318 58341
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58318) (leftOrdinal := 0)
    (rightResult := 58341) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15370⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58345

namespace LeftMerge58362
def owner : Owner := ⟨.program ⟨214⟩, ⟨20687⟩⟩
def mergeEvent : Nat := 58362
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58359RawTerms
def group : MergeGroup := .relation 58361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58361) (rhsResult := 58359)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58360 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩) (none) 58359) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58362

namespace LeftMerge58363
def owner : Owner := ⟨.program ⟨214⟩, ⟨20687⟩⟩
def mergeEvent : Nat := 58363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58359RawTerms
def group : MergeGroup := .relation 58361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58361) (rhsResult := 58359)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58360 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩) (none) 58359) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58363

namespace LeftMerge58364
def owner : Owner := ⟨.program ⟨214⟩, ⟨20687⟩⟩
def mergeEvent : Nat := 58364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23850⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58359RawTerms
def group : MergeGroup := .relation 58361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58361) (rhsResult := 58359)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58360 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩) (none) 58359) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23850⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨23850⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58364

namespace LeftMerge58365
def owner : Owner := ⟨.program ⟨214⟩, ⟨20687⟩⟩
def mergeEvent : Nat := 58365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15370⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58359RawTerms
def group : MergeGroup := .relation 58361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58361) (rhsResult := 58359)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58360 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩) (none) 58359) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15370⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58365

namespace LeftMerge58370
def owner : Owner := ⟨.program ⟨214⟩, ⟨26797⟩⟩
def mergeEvent : Nat := 58370
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58366RawTerms
def rightRaw : List Term := Proof.Events227.exact58188RawTerms
def group : MergeGroup := .operator 58366 58188
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58366) (leftOrdinal := 0)
    (rightResult := 58188) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26794⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58370

namespace LeftMerge58371
def owner : Owner := ⟨.program ⟨214⟩, ⟨26797⟩⟩
def mergeEvent : Nat := 58371
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23850⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58366RawTerms
def rightRaw : List Term := Proof.Events227.exact58188RawTerms
def group : MergeGroup := .operator 58366 58188
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58366) (leftOrdinal := 2)
    (rightResult := 58188) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23850⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23850⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15118⟩⟩], [⟨.program ⟨214⟩, ⟨23850⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58371

namespace LeftMerge58397
def owner : Owner := ⟨.program ⟨214⟩, ⟨10687⟩⟩
def mergeEvent : Nat := 58397
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events010.exact2706RawTerms
def rightRaw : List Term := Proof.Events197.exact50670RawTerms
def group : MergeGroup := .operator 2706 50670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2706) (leftOrdinal := 0)
    (rightResult := 50670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58397

namespace LeftMerge58402
def owner : Owner := ⟨.program ⟨214⟩, ⟨7267⟩⟩
def mergeEvent : Nat := 58402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50540RawTerms
def rightRaw : List Term := Proof.Events056.exact14488RawTerms
def group : MergeGroup := .operator 50540 14488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50540) (leftOrdinal := 0)
    (rightResult := 14488) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58402

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
