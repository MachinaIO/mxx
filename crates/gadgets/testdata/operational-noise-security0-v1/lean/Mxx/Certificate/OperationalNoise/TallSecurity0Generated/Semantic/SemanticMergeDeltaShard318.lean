import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge52159
def owner : Owner := ⟨.program ⟨214⟩, ⟨10036⟩⟩
def mergeEvent : Nat := 52159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2410RawTerms
def rightRaw : List Term := Proof.Events197.exact50670RawTerms
def group : MergeGroup := .operator 2410 50670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2410) (leftOrdinal := 0)
    (rightResult := 50670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10035⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52159

namespace LeftMerge52164
def owner : Owner := ⟨.program ⟨214⟩, ⟨7261⟩⟩
def mergeEvent : Nat := 52164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50540RawTerms
def rightRaw : List Term := Proof.Events031.exact8016RawTerms
def group : MergeGroup := .operator 50540 8016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50540) (leftOrdinal := 0)
    (rightResult := 8016) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52164

namespace LeftMerge52181
def owner : Owner := ⟨.program ⟨214⟩, ⟨10039⟩⟩
def mergeEvent : Nat := 52181
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52175RawTerms
def rightRaw : List Term := Proof.Events031.exact8005RawTerms
def group : MergeGroup := .operator 52175 8005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52175) (leftOrdinal := 1)
    (rightResult := 8005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52181

namespace LeftMerge52183
def owner : Owner := ⟨.program ⟨214⟩, ⟨10039⟩⟩
def mergeEvent : Nat := 52183
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def rhsRaw : List Term := Proof.Events031.exact7975RawTerms
def group : MergeGroup := .relation 52182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52182) (rhsResult := 7975)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7873⟩⟩) ⟨6787⟩ 7975) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52183

namespace LeftMerge52184
def owner : Owner := ⟨.program ⟨214⟩, ⟨10039⟩⟩
def mergeEvent : Nat := 52184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52175RawTerms
def rightRaw : List Term := Proof.Events031.exact8005RawTerms
def group : MergeGroup := .operator 52175 8005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52175) (leftOrdinal := 0)
    (rightResult := 8005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52184

namespace LeftMerge52189
def owner : Owner := ⟨.program ⟨214⟩, ⟨12777⟩⟩
def mergeEvent : Nat := 52189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52185RawTerms
def rightRaw : List Term := Proof.Events203.exact52155RawTerms
def group : MergeGroup := .operator 52185 52155
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52185) (leftOrdinal := 1)
    (rightResult := 52155) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52189

namespace LeftMerge52197
def owner : Owner := ⟨.program ⟨214⟩, ⟨25533⟩⟩
def mergeEvent : Nat := 52197
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52191RawTerms
def rightRaw : List Term := Proof.Events203.exact52127RawTerms
def group : MergeGroup := .operator 52191 52127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52191) (leftOrdinal := 1)
    (rightResult := 52127) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25532⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52197

namespace LeftMerge52199
def owner : Owner := ⟨.program ⟨214⟩, ⟨25533⟩⟩
def mergeEvent : Nat := 52199
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23292⟩⟩] } }
def rhsRaw : List Term := Proof.Events203.exact52124RawTerms
def group : MergeGroup := .relation 52198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52198) (rhsResult := 52124)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25532⟩⟩) ⟨23292⟩ 52124) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23292⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52199

namespace LeftMerge52200
def owner : Owner := ⟨.program ⟨214⟩, ⟨25533⟩⟩
def mergeEvent : Nat := 52200
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52191RawTerms
def rightRaw : List Term := Proof.Events203.exact52127RawTerms
def group : MergeGroup := .operator 52191 52127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52191) (leftOrdinal := 0)
    (rightResult := 52127) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25532⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52200

namespace LeftMerge52214
def owner : Owner := ⟨.program ⟨214⟩, ⟨20039⟩⟩
def mergeEvent : Nat := 52214
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20036⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events203.exact52208RawTerms
def group : MergeGroup := .operator 50762 52208
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 52208) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20036⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20036⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52214

namespace LeftMerge52293
def owner : Owner := ⟨.program ⟨214⟩, ⟨12771⟩⟩
def mergeEvent : Nat := 52293
def frameStart : Nat := 52263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events204.exact52289RawTerms
def rightRaw : List Term := Proof.Events204.exact52286RawTerms
def group : MergeGroup := .operator 52289 52286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52289) (leftOrdinal := 0)
    (rightResult := 52286) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10035⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52293

namespace LeftMerge52323
def owner : Owner := ⟨.program ⟨214⟩, ⟨12864⟩⟩
def mergeEvent : Nat := 52323
def frameStart : Nat := 52263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events204.exact52319RawTerms
def rightRaw : List Term := Proof.Events204.exact52317RawTerms
def group : MergeGroup := .operator 52319 52317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52319) (leftOrdinal := 0)
    (rightResult := 52317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52323

namespace LeftMerge52346
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def mergeEvent : Nat := 52346
def frameStart : Nat := 52263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events204.exact52342RawTerms
def rightRaw : List Term := Proof.Events204.exact52339RawTerms
def group : MergeGroup := .operator 52342 52339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52342) (leftOrdinal := 0)
    (rightResult := 52339) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52346

namespace LeftMerge52355
def owner : Owner := ⟨.program ⟨214⟩, ⟨25535⟩⟩
def mergeEvent : Nat := 52355
def frameStart : Nat := 52263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩] } }
def leftRaw : List Term := Proof.Events204.exact52351RawTerms
def rightRaw : List Term := Proof.Events204.exact52308RawTerms
def group : MergeGroup := .operator 52351 52308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52351) (leftOrdinal := 0)
    (rightResult := 52308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52355

namespace LeftMerge52356
def owner : Owner := ⟨.program ⟨214⟩, ⟨25535⟩⟩
def mergeEvent : Nat := 52356
def frameStart : Nat := 52263
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩] } }
def leftRaw : List Term := Proof.Events204.exact52351RawTerms
def rightRaw : List Term := Proof.Events204.exact52308RawTerms
def group : MergeGroup := .operator 52351 52308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52351) (leftOrdinal := 1)
    (rightResult := 52308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25532⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52356

namespace LeftMerge52358
def owner : Owner := ⟨.program ⟨214⟩, ⟨25535⟩⟩
def mergeEvent : Nat := 52358
def frameStart : Nat := 52263
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23292⟩⟩] } }
def rhsRaw : List Term := Proof.Events204.exact52305RawTerms
def group : MergeGroup := .relation 52357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52357) (rhsResult := 52305)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25532⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25532⟩⟩) ⟨23292⟩ 52305) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23292⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10035⟩⟩, ⟨.program ⟨214⟩, ⟨12770⟩⟩], [⟨.program ⟨214⟩, ⟨23292⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52358

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
