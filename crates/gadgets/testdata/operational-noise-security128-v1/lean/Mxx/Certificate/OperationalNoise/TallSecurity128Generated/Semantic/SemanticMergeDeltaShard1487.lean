import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge241384
def owner : Owner := ⟨.program ⟨257⟩, ⟨63352⟩⟩
def mergeEvent : Nat := 241384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩] } }
def rhsRaw : List Term := Proof.Events942.exact241380RawTerms
def group : MergeGroup := .relation 241382
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241382) (rhsResult := 241380)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241381 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩]⟩) (none) 241380) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241384

namespace LeftMerge241385
def owner : Owner := ⟨.program ⟨257⟩, ⟨63352⟩⟩
def mergeEvent : Nat := 241385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63917⟩⟩] } }
def rhsRaw : List Term := Proof.Events942.exact241380RawTerms
def group : MergeGroup := .relation 241382
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241382) (rhsResult := 241380)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241381 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩]⟩) (none) 241380) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63917⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨63917⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241385

namespace LeftMerge241386
def owner : Owner := ⟨.program ⟨257⟩, ⟨63352⟩⟩
def mergeEvent : Nat := 241386
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events942.exact241380RawTerms
def group : MergeGroup := .relation 241382
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241382) (rhsResult := 241380)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241381 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63349⟩⟩]⟩) (none) 241380) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241386

namespace LeftMerge241391
def owner : Owner := ⟨.program ⟨257⟩, ⟨64419⟩⟩
def mergeEvent : Nat := 241391
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63917⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241387RawTerms
def rightRaw : List Term := Proof.Events942.exact241201RawTerms
def group : MergeGroup := .operator 241387 241201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241387) (leftOrdinal := 2)
    (rightResult := 241201) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63917⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63917⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25466⟩⟩, ⟨.program ⟨257⟩, ⟨62411⟩⟩], [⟨.program ⟨257⟩, ⟨63917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241391

namespace LeftMerge241392
def owner : Owner := ⟨.program ⟨257⟩, ⟨64419⟩⟩
def mergeEvent : Nat := 241392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241387RawTerms
def rightRaw : List Term := Proof.Events942.exact241201RawTerms
def group : MergeGroup := .operator 241387 241201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241387) (leftOrdinal := 1)
    (rightResult := 241201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64417⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241392

namespace LeftMerge241400
def owner : Owner := ⟨.program ⟨257⟩, ⟨64812⟩⟩
def mergeEvent : Nat := 241400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241394RawTerms
def rightRaw : List Term := Proof.Events941.exact241117RawTerms
def group : MergeGroup := .operator 241394 241117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241394) (leftOrdinal := 0)
    (rightResult := 241117) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64810⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241400

namespace LeftMerge241401
def owner : Owner := ⟨.program ⟨257⟩, ⟨64812⟩⟩
def mergeEvent : Nat := 241401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩] } }
def leftRaw : List Term := Proof.Events942.exact241394RawTerms
def rightRaw : List Term := Proof.Events941.exact241117RawTerms
def group : MergeGroup := .operator 241394 241117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241394) (leftOrdinal := 1)
    (rightResult := 241117) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64810⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241401

namespace LeftMerge241403
def owner : Owner := ⟨.program ⟨257⟩, ⟨64812⟩⟩
def mergeEvent : Nat := 241403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64063⟩⟩] } }
def rhsRaw : List Term := Proof.Events941.exact241114RawTerms
def group : MergeGroup := .relation 241402
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241402) (rhsResult := 241114)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64810⟩⟩) ⟨64063⟩ 241114) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64063⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨62792⟩⟩], [⟨.program ⟨257⟩, ⟨64063⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241403

namespace LeftMerge241417
def owner : Owner := ⟨.program ⟨257⟩, ⟨63639⟩⟩
def mergeEvent : Nat := 241417
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63636⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events943.exact241411RawTerms
def group : MergeGroup := .operator 236870 241411
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 241411) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63636⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63636⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241417

namespace LeftMerge241538
def owner : Owner := ⟨.program ⟨257⟩, ⟨64280⟩⟩
def mergeEvent : Nat := 241538
def frameStart : Nat := 241472
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events943.exact241534RawTerms
def rightRaw : List Term := Proof.Events943.exact241532RawTerms
def group : MergeGroup := .operator 241534 241532
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241534) (leftOrdinal := 0)
    (rightResult := 241532) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241538

namespace LeftMerge241550
def owner : Owner := ⟨.program ⟨257⟩, ⟨64811⟩⟩
def mergeEvent : Nat := 241550
def frameStart : Nat := 241472
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩] } }
def leftRaw : List Term := Proof.Events943.exact241546RawTerms
def rightRaw : List Term := Proof.Events943.exact241523RawTerms
def group : MergeGroup := .operator 241546 241523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241546) (leftOrdinal := 0)
    (rightResult := 241523) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64810⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241550

namespace LeftMerge241551
def owner : Owner := ⟨.program ⟨257⟩, ⟨64811⟩⟩
def mergeEvent : Nat := 241551
def frameStart : Nat := 241472
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩] } }
def leftRaw : List Term := Proof.Events943.exact241546RawTerms
def rightRaw : List Term := Proof.Events943.exact241523RawTerms
def group : MergeGroup := .operator 241546 241523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241546) (leftOrdinal := 1)
    (rightResult := 241523) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64810⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241551

namespace LeftMerge241553
def owner : Owner := ⟨.program ⟨257⟩, ⟨64811⟩⟩
def mergeEvent : Nat := 241553
def frameStart : Nat := 241472
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64063⟩⟩] } }
def rhsRaw : List Term := Proof.Events943.exact241520RawTerms
def group : MergeGroup := .relation 241552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241552) (rhsResult := 241520)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64810⟩⟩) ⟨64063⟩ 241520) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64063⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62792⟩⟩], [⟨.program ⟨257⟩, ⟨64063⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241553

namespace LeftMerge241561
def owner : Owner := ⟨.program ⟨257⟩, ⟨63045⟩⟩
def mergeEvent : Nat := 241561
def frameStart : Nat := 241472
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63043⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events943.exact241534RawTerms
def rightRaw : List Term := Proof.Events943.exact241557RawTerms
def group : MergeGroup := .operator 241534 241557
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241534) (leftOrdinal := 0)
    (rightResult := 241557) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63043⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63043⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241561

namespace LeftMerge241578
def owner : Owner := ⟨.program ⟨257⟩, ⟨63639⟩⟩
def mergeEvent : Nat := 241578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }
def rhsRaw : List Term := Proof.Events943.exact241575RawTerms
def group : MergeGroup := .relation 241577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241577) (rhsResult := 241575)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63636⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241576 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63636⟩⟩]⟩) (none) 241575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241578

namespace LeftMerge241579
def owner : Owner := ⟨.program ⟨257⟩, ⟨63639⟩⟩
def mergeEvent : Nat := 241579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩] } }
def rhsRaw : List Term := Proof.Events943.exact241575RawTerms
def group : MergeGroup := .relation 241577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 241577) (rhsResult := 241575)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63636⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 241576 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63636⟩⟩]⟩) (none) 241575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64810⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge241579

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
