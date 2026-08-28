import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge241181
def owner : Owner := ⟨.program ⟨257⟩, ⟨62418⟩⟩
def mergeEvent : Nat := 241181
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241175RawTerms
def rightRaw : List Term := Proof.Events084.exact21619RawTerms
def group : MergeGroup := .operator 241175 21619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241175) (leftOrdinal := 1)
    (rightResult := 21619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241181

namespace LeftMerge241183
def owner : Owner := ⟨.program ⟨257⟩, ⟨62418⟩⟩
def mergeEvent : Nat := 241183
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }
def rhsRaw : List Term := Proof.Events084.exact21589RawTerms
def group : MergeGroup := .relation 241182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241182) (rhsResult := 21589)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9538⟩⟩) ⟨7275⟩ 21589) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241183

namespace LeftMerge241184
def owner : Owner := ⟨.program ⟨257⟩, ⟨62418⟩⟩
def mergeEvent : Nat := 241184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241175RawTerms
def rightRaw : List Term := Proof.Events084.exact21619RawTerms
def group : MergeGroup := .operator 241175 21619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241175) (leftOrdinal := 0)
    (rightResult := 21619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241184

namespace LeftMerge241189
def owner : Owner := ⟨.program ⟨257⟩, ⟨62419⟩⟩
def mergeEvent : Nat := 241189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241185RawTerms
def rightRaw : List Term := Proof.Events942.exact241155RawTerms
def group : MergeGroup := .operator 241185 241155
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241185) (leftOrdinal := 1)
    (rightResult := 241155) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241189

namespace LeftMerge241197
def owner : Owner := ⟨.program ⟨257⟩, ⟨64418⟩⟩
def mergeEvent : Nat := 241197
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241191RawTerms
def rightRaw : List Term := Proof.Events941.exact241127RawTerms
def group : MergeGroup := .operator 241191 241127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241191) (leftOrdinal := 1)
    (rightResult := 241127) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64417⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241197

namespace LeftMerge241199
def owner : Owner := ⟨.program ⟨257⟩, ⟨64418⟩⟩
def mergeEvent : Nat := 241199
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63917⟩⟩] } }
def rhsRaw : List Term := Proof.Events941.exact241124RawTerms
def group : MergeGroup := .relation 241198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241198) (rhsResult := 241124)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64417⟩⟩) ⟨63917⟩ 241124) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63917⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨63917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241199

namespace LeftMerge241200
def owner : Owner := ⟨.program ⟨257⟩, ⟨64418⟩⟩
def mergeEvent : Nat := 241200
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241191RawTerms
def rightRaw : List Term := Proof.Events941.exact241127RawTerms
def group : MergeGroup := .operator 241191 241127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241191) (leftOrdinal := 0)
    (rightResult := 241127) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64417⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241200

namespace LeftMerge241214
def owner : Owner := ⟨.program ⟨257⟩, ⟨63352⟩⟩
def mergeEvent : Nat := 241214
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events942.exact241208RawTerms
def group : MergeGroup := .operator 236870 241208
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 241208) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241214

namespace LeftMerge241293
def owner : Owner := ⟨.program ⟨257⟩, ⟨62412⟩⟩
def mergeEvent : Nat := 241293
def frameStart : Nat := 241263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events942.exact241289RawTerms
def rightRaw : List Term := Proof.Events942.exact241286RawTerms
def group : MergeGroup := .operator 241289 241286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241289) (leftOrdinal := 0)
    (rightResult := 241286) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25466⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241293

namespace LeftMerge241323
def owner : Owner := ⟨.program ⟨257⟩, ⟨64200⟩⟩
def mergeEvent : Nat := 241323
def frameStart : Nat := 241263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241319RawTerms
def rightRaw : List Term := Proof.Events942.exact241317RawTerms
def group : MergeGroup := .operator 241319 241317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241319) (leftOrdinal := 0)
    (rightResult := 241317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241323

namespace LeftMerge241346
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 241346
def frameStart : Nat := 241263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241342RawTerms
def rightRaw : List Term := Proof.Events942.exact241339RawTerms
def group : MergeGroup := .operator 241342 241339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241342) (leftOrdinal := 0)
    (rightResult := 241339) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241346

namespace LeftMerge241355
def owner : Owner := ⟨.program ⟨257⟩, ⟨64420⟩⟩
def mergeEvent : Nat := 241355
def frameStart : Nat := 241263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241351RawTerms
def rightRaw : List Term := Proof.Events942.exact241308RawTerms
def group : MergeGroup := .operator 241351 241308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241351) (leftOrdinal := 0)
    (rightResult := 241308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64417⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241355

namespace LeftMerge241356
def owner : Owner := ⟨.program ⟨257⟩, ⟨64420⟩⟩
def mergeEvent : Nat := 241356
def frameStart : Nat := 241263
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241351RawTerms
def rightRaw : List Term := Proof.Events942.exact241308RawTerms
def group : MergeGroup := .operator 241351 241308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241351) (leftOrdinal := 1)
    (rightResult := 241308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64417⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241356

namespace LeftMerge241358
def owner : Owner := ⟨.program ⟨257⟩, ⟨64420⟩⟩
def mergeEvent : Nat := 241358
def frameStart : Nat := 241263
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63917⟩⟩] } }
def rhsRaw : List Term := Proof.Events942.exact241305RawTerms
def group : MergeGroup := .relation 241357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241357) (rhsResult := 241305)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64417⟩⟩) ⟨63917⟩ 241305) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63917⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨63917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241358

namespace LeftMerge241366
def owner : Owner := ⟨.program ⟨257⟩, ⟨62794⟩⟩
def mergeEvent : Nat := 241366
def frameStart : Nat := 241263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241319RawTerms
def rightRaw : List Term := Proof.Events942.exact241362RawTerms
def group : MergeGroup := .operator 241319 241362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241319) (leftOrdinal := 0)
    (rightResult := 241362) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241366

namespace LeftMerge241383
def owner : Owner := ⟨.program ⟨257⟩, ⟨63352⟩⟩
def mergeEvent : Nat := 241383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events942.exact241380RawTerms
def group : MergeGroup := .relation 241382
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241382) (rhsResult := 241380)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241381 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩]⟩) (none) 241380) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241383

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
