import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge30189
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16807⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21392RawTerms
def group : MergeGroup := .relation 30188
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30188) (rhsResult := 21392)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16807⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16807⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30189

namespace LeftMerge30190
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30190
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 14)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30190

namespace LeftMerge30191
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30191
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 27)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30191

namespace LeftMerge30193
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30193
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21392RawTerms
def group : MergeGroup := .relation 30192
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30192) (rhsResult := 21392)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30193

namespace LeftMerge30194
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30194
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 13)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30194

namespace LeftMerge30195
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30195
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 34)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30195

namespace LeftMerge30197
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30197
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21392RawTerms
def group : MergeGroup := .relation 30196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30196) (rhsResult := 21392)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30197

namespace LeftMerge30198
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30198
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 12)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30198

namespace LeftMerge30199
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30199
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 32)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30199

namespace LeftMerge30201
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30201
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21392RawTerms
def group : MergeGroup := .relation 30200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30200) (rhsResult := 21392)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30201

namespace LeftMerge30202
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30202
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 11)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30202

namespace LeftMerge30203
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30203
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 30)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30203

namespace LeftMerge30205
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21392RawTerms
def group : MergeGroup := .relation 30204
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30204) (rhsResult := 21392)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30205

namespace LeftMerge30206
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 10)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge30206

namespace LeftMerge30207
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events117.exact30172RawTerms
def rightRaw : List Term := Proof.Events083.exact21395RawTerms
def group : MergeGroup := .operator 30172 21395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 30172) (leftOrdinal := 26)
    (rightResult := 21395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30207

namespace LeftMerge30209
def owner : Owner := ⟨.program ⟨214⟩, ⟨30188⟩⟩
def mergeEvent : Nat := 30209
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21392RawTerms
def group : MergeGroup := .relation 30208
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 30208) (rhsResult := 21392)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18624⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge30209

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
