import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge206150
def owner : Owner := ⟨.program ⟨257⟩, ⟨51795⟩⟩
def mergeEvent : Nat := 206150
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events805.exact206144RawTerms
def group : MergeGroup := .relation 206146
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 206146) (rhsResult := 206144)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51792⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 206145 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51792⟩⟩]⟩) (none) 206144) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51203⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge206150

namespace LeftMerge206155
def owner : Owner := ⟨.program ⟨257⟩, ⟨53010⟩⟩
def mergeEvent : Nat := 206155
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53007⟩⟩] } }
def leftRaw : List Term := Proof.Events805.exact206151RawTerms
def rightRaw : List Term := Proof.Events804.exact205973RawTerms
def group : MergeGroup := .operator 206151 205973
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 206151) (leftOrdinal := 0)
    (rightResult := 205973) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53007⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53007⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53007⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge206155

namespace LeftMerge206156
def owner : Owner := ⟨.program ⟨257⟩, ⟨53010⟩⟩
def mergeEvent : Nat := 206156
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52178⟩⟩] } }
def leftRaw : List Term := Proof.Events805.exact206151RawTerms
def rightRaw : List Term := Proof.Events804.exact205973RawTerms
def group : MergeGroup := .operator 206151 205973
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 206151) (leftOrdinal := 2)
    (rightResult := 205973) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52178⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52178⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨52178⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge206156

namespace LeftMerge206164
def owner : Owner := ⟨.program ⟨257⟩, ⟨53011⟩⟩
def mergeEvent : Nat := 206164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }
def leftRaw : List Term := Proof.Events805.exact206158RawTerms
def rightRaw : List Term := Proof.Events061.exact15802RawTerms
def group : MergeGroup := .operator 206158 15802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 206158) (leftOrdinal := 0)
    (rightResult := 15802) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge206164

namespace LeftMerge206165
def owner : Owner := ⟨.program ⟨257⟩, ⟨53011⟩⟩
def mergeEvent : Nat := 206165
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }
def leftRaw : List Term := Proof.Events805.exact206158RawTerms
def rightRaw : List Term := Proof.Events061.exact15802RawTerms
def group : MergeGroup := .operator 206158 15802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 206158) (leftOrdinal := 1)
    (rightResult := 15802) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge206165

namespace LeftMerge206167
def owner : Owner := ⟨.program ⟨257⟩, ⟨53011⟩⟩
def mergeEvent : Nat := 206167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15795RawTerms
def group : MergeGroup := .relation 206166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 206166) (rhsResult := 15795)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7131⟩⟩) ⟨7031⟩ 15795) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge206167

namespace LeftMerge206181
def owner : Owner := ⟨.program ⟨257⟩, ⟨33949⟩⟩
def mergeEvent : Nat := 206181
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199929RawTerms
def rightRaw : List Term := Proof.Events805.exact206175RawTerms
def group : MergeGroup := .operator 199929 206175
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199929) (leftOrdinal := 0)
    (rightResult := 206175) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33947⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge206181

namespace LeftMerge206182
def owner : Owner := ⟨.program ⟨257⟩, ⟨33949⟩⟩
def mergeEvent : Nat := 206182
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩] } }
def leftRaw : List Term := Proof.Events780.exact199929RawTerms
def rightRaw : List Term := Proof.Events805.exact206175RawTerms
def group : MergeGroup := .operator 199929 206175
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199929) (leftOrdinal := 1)
    (rightResult := 206175) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33947⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge206182

namespace LeftMerge206184
def owner : Owner := ⟨.program ⟨257⟩, ⟨33949⟩⟩
def mergeEvent : Nat := 206184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33118⟩⟩] } }
def rhsRaw : List Term := Proof.Events805.exact206172RawTerms
def group : MergeGroup := .relation 206183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 206183) (rhsResult := 206172)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33947⟩⟩) ⟨33118⟩ 206172) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33118⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨33118⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge206184

namespace LeftMerge206198
def owner : Owner := ⟨.program ⟨257⟩, ⟨32735⟩⟩
def mergeEvent : Nat := 206198
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32732⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events805.exact206192RawTerms
def group : MergeGroup := .operator 192995 206192
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 206192) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32732⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32732⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge206198

namespace LeftMerge206319
def owner : Owner := ⟨.program ⟨257⟩, ⟨33316⟩⟩
def mergeEvent : Nat := 206319
def frameStart : Nat := 206253
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events805.exact206315RawTerms
def rightRaw : List Term := Proof.Events805.exact206313RawTerms
def group : MergeGroup := .operator 206315 206313
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 206315) (leftOrdinal := 0)
    (rightResult := 206313) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge206319

namespace LeftMerge206331
def owner : Owner := ⟨.program ⟨257⟩, ⟨33948⟩⟩
def mergeEvent : Nat := 206331
def frameStart : Nat := 206253
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩] } }
def leftRaw : List Term := Proof.Events805.exact206327RawTerms
def rightRaw : List Term := Proof.Events805.exact206304RawTerms
def group : MergeGroup := .operator 206327 206304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 206327) (leftOrdinal := 0)
    (rightResult := 206304) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33947⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge206331

namespace LeftMerge206332
def owner : Owner := ⟨.program ⟨257⟩, ⟨33948⟩⟩
def mergeEvent : Nat := 206332
def frameStart : Nat := 206253
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩] } }
def leftRaw : List Term := Proof.Events805.exact206327RawTerms
def rightRaw : List Term := Proof.Events805.exact206304RawTerms
def group : MergeGroup := .operator 206327 206304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 206327) (leftOrdinal := 1)
    (rightResult := 206304) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33947⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge206332

namespace LeftMerge206334
def owner : Owner := ⟨.program ⟨257⟩, ⟨33948⟩⟩
def mergeEvent : Nat := 206334
def frameStart : Nat := 206253
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33118⟩⟩] } }
def rhsRaw : List Term := Proof.Events805.exact206301RawTerms
def group : MergeGroup := .relation 206333
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 206333) (rhsResult := 206301)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33947⟩⟩) ⟨33118⟩ 206301) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33118⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨33118⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge206334

namespace LeftMerge206342
def owner : Owner := ⟨.program ⟨257⟩, ⟨32142⟩⟩
def mergeEvent : Nat := 206342
def frameStart : Nat := 206253
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32139⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events805.exact206315RawTerms
def rightRaw : List Term := Proof.Events806.exact206338RawTerms
def group : MergeGroup := .operator 206315 206338
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 206315) (leftOrdinal := 0)
    (rightResult := 206338) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32139⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge206342

namespace LeftMerge206359
def owner : Owner := ⟨.program ⟨257⟩, ⟨32735⟩⟩
def mergeEvent : Nat := 206359
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }
def rhsRaw : List Term := Proof.Events806.exact206356RawTerms
def group : MergeGroup := .relation 206358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 206358) (rhsResult := 206356)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32732⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 206357 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32732⟩⟩]⟩) (none) 206356) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge206359

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
