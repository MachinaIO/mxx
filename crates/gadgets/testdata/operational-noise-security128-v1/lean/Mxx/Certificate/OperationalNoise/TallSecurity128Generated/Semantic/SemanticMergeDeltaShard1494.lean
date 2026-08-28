import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge242630
def owner : Owner := ⟨.program ⟨257⟩, ⟨53478⟩⟩
def mergeEvent : Nat := 242630
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events947.exact242621RawTerms
def rightRaw : List Term := Proof.Events090.exact23122RawTerms
def group : MergeGroup := .operator 242621 23122
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242621) (leftOrdinal := 0)
    (rightResult := 23122) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242630

namespace LeftMerge242635
def owner : Owner := ⟨.program ⟨257⟩, ⟨53479⟩⟩
def mergeEvent : Nat := 242635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }
def leftRaw : List Term := Proof.Events947.exact242631RawTerms
def rightRaw : List Term := Proof.Events947.exact242601RawTerms
def group : MergeGroup := .operator 242631 242601
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242631) (leftOrdinal := 1)
    (rightResult := 242601) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242635

namespace LeftMerge242643
def owner : Owner := ⟨.program ⟨257⟩, ⟨55478⟩⟩
def mergeEvent : Nat := 242643
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩] } }
def leftRaw : List Term := Proof.Events947.exact242637RawTerms
def rightRaw : List Term := Proof.Events947.exact242573RawTerms
def group : MergeGroup := .operator 242637 242573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242637) (leftOrdinal := 1)
    (rightResult := 242573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55477⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242643

namespace LeftMerge242645
def owner : Owner := ⟨.program ⟨257⟩, ⟨55478⟩⟩
def mergeEvent : Nat := 242645
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54977⟩⟩] } }
def rhsRaw : List Term := Proof.Events947.exact242570RawTerms
def group : MergeGroup := .relation 242644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242644) (rhsResult := 242570)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55477⟩⟩) ⟨54977⟩ 242570) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54977⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨54977⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242645

namespace LeftMerge242646
def owner : Owner := ⟨.program ⟨257⟩, ⟨55478⟩⟩
def mergeEvent : Nat := 242646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩] } }
def leftRaw : List Term := Proof.Events947.exact242637RawTerms
def rightRaw : List Term := Proof.Events947.exact242573RawTerms
def group : MergeGroup := .operator 242637 242573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242637) (leftOrdinal := 0)
    (rightResult := 242573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55477⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242646

namespace LeftMerge242660
def owner : Owner := ⟨.program ⟨257⟩, ⟨54412⟩⟩
def mergeEvent : Nat := 242660
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events947.exact242654RawTerms
def group : MergeGroup := .operator 236870 242654
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 242654) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54409⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242660

namespace LeftMerge242739
def owner : Owner := ⟨.program ⟨257⟩, ⟨53472⟩⟩
def mergeEvent : Nat := 242739
def frameStart : Nat := 242709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events948.exact242735RawTerms
def rightRaw : List Term := Proof.Events948.exact242732RawTerms
def group : MergeGroup := .operator 242735 242732
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242735) (leftOrdinal := 0)
    (rightResult := 242732) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242739

namespace LeftMerge242769
def owner : Owner := ⟨.program ⟨257⟩, ⟨55260⟩⟩
def mergeEvent : Nat := 242769
def frameStart : Nat := 242709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events948.exact242765RawTerms
def rightRaw : List Term := Proof.Events948.exact242763RawTerms
def group : MergeGroup := .operator 242765 242763
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242765) (leftOrdinal := 0)
    (rightResult := 242763) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242769

namespace LeftMerge242792
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 242792
def frameStart : Nat := 242709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events948.exact242788RawTerms
def rightRaw : List Term := Proof.Events948.exact242785RawTerms
def group : MergeGroup := .operator 242788 242785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242788) (leftOrdinal := 0)
    (rightResult := 242785) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242792

namespace LeftMerge242801
def owner : Owner := ⟨.program ⟨257⟩, ⟨55480⟩⟩
def mergeEvent : Nat := 242801
def frameStart : Nat := 242709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩] } }
def leftRaw : List Term := Proof.Events948.exact242797RawTerms
def rightRaw : List Term := Proof.Events948.exact242754RawTerms
def group : MergeGroup := .operator 242797 242754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242797) (leftOrdinal := 0)
    (rightResult := 242754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55477⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242801

namespace LeftMerge242802
def owner : Owner := ⟨.program ⟨257⟩, ⟨55480⟩⟩
def mergeEvent : Nat := 242802
def frameStart : Nat := 242709
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩] } }
def leftRaw : List Term := Proof.Events948.exact242797RawTerms
def rightRaw : List Term := Proof.Events948.exact242754RawTerms
def group : MergeGroup := .operator 242797 242754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242797) (leftOrdinal := 1)
    (rightResult := 242754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55477⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242802

namespace LeftMerge242804
def owner : Owner := ⟨.program ⟨257⟩, ⟨55480⟩⟩
def mergeEvent : Nat := 242804
def frameStart : Nat := 242709
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54977⟩⟩] } }
def rhsRaw : List Term := Proof.Events948.exact242751RawTerms
def group : MergeGroup := .relation 242803
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242803) (rhsResult := 242751)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55477⟩⟩) ⟨54977⟩ 242751) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54977⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨54977⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242804

namespace LeftMerge242812
def owner : Owner := ⟨.program ⟨257⟩, ⟨53854⟩⟩
def mergeEvent : Nat := 242812
def frameStart : Nat := 242709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events948.exact242765RawTerms
def rightRaw : List Term := Proof.Events948.exact242808RawTerms
def group : MergeGroup := .operator 242765 242808
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242765) (leftOrdinal := 0)
    (rightResult := 242808) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242812

namespace LeftMerge242829
def owner : Owner := ⟨.program ⟨257⟩, ⟨54412⟩⟩
def mergeEvent : Nat := 242829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events948.exact242826RawTerms
def group : MergeGroup := .relation 242828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242828) (rhsResult := 242826)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242827 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩) (none) 242826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242829

namespace LeftMerge242830
def owner : Owner := ⟨.program ⟨257⟩, ⟨54412⟩⟩
def mergeEvent : Nat := 242830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩] } }
def rhsRaw : List Term := Proof.Events948.exact242826RawTerms
def group : MergeGroup := .relation 242828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242828) (rhsResult := 242826)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242827 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩) (none) 242826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242830

namespace LeftMerge242831
def owner : Owner := ⟨.program ⟨257⟩, ⟨54412⟩⟩
def mergeEvent : Nat := 242831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54977⟩⟩] } }
def rhsRaw : List Term := Proof.Events948.exact242826RawTerms
def group : MergeGroup := .relation 242828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242828) (rhsResult := 242826)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242827 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩) (none) 242826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54977⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨54977⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242831

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
