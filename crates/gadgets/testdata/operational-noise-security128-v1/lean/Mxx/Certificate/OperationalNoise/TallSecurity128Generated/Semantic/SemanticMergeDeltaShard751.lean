import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge124104
def owner : Owner := ⟨.program ⟨257⟩, ⟨69864⟩⟩
def mergeEvent : Nat := 124104
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69861⟩⟩] } }
def leftRaw : List Term := Proof.Events484.exact124100RawTerms
def rightRaw : List Term := Proof.Events484.exact123922RawTerms
def group : MergeGroup := .operator 124100 123922
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124100) (leftOrdinal := 0)
    (rightResult := 123922) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69861⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69861⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124104

namespace LeftMerge124105
def owner : Owner := ⟨.program ⟨257⟩, ⟨69864⟩⟩
def mergeEvent : Nat := 124105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨65756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68646⟩⟩] } }
def leftRaw : List Term := Proof.Events484.exact124100RawTerms
def rightRaw : List Term := Proof.Events484.exact123922RawTerms
def group : MergeGroup := .operator 124100 123922
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124100) (leftOrdinal := 2)
    (rightResult := 123922) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨65756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68646⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨65756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68646⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨65756⟩⟩], [⟨.program ⟨257⟩, ⟨68646⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124105

namespace LeftMerge124131
def owner : Owner := ⟨.program ⟨257⟩, ⟨25443⟩⟩
def mergeEvent : Nat := 124131
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events021.exact5537RawTerms
def rightRaw : List Term := Proof.Events467.exact119778RawTerms
def group : MergeGroup := .operator 5537 119778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5537) (leftOrdinal := 0)
    (rightResult := 119778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25442⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124131

namespace LeftMerge124136
def owner : Owner := ⟨.program ⟨257⟩, ⟨8125⟩⟩
def mergeEvent : Nat := 124136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }
def leftRaw : List Term := Proof.Events467.exact119648RawTerms
def rightRaw : List Term := Proof.Events084.exact21589RawTerms
def group : MergeGroup := .operator 119648 21589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119648) (leftOrdinal := 0)
    (rightResult := 21589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124136

namespace LeftMerge124153
def owner : Owner := ⟨.program ⟨257⟩, ⟨62360⟩⟩
def mergeEvent : Nat := 124153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events484.exact124147RawTerms
def rightRaw : List Term := Proof.Events021.exact5540RawTerms
def group : MergeGroup := .operator 124147 5540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124147) (leftOrdinal := 1)
    (rightResult := 5540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124153

namespace LeftMerge124154
def owner : Owner := ⟨.program ⟨257⟩, ⟨62360⟩⟩
def mergeEvent : Nat := 124154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }
def leftRaw : List Term := Proof.Events484.exact124147RawTerms
def rightRaw : List Term := Proof.Events021.exact5540RawTerms
def group : MergeGroup := .operator 124147 5540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124147) (leftOrdinal := 0)
    (rightResult := 5540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124154

namespace LeftMerge124159
def owner : Owner := ⟨.program ⟨257⟩, ⟨62361⟩⟩
def mergeEvent : Nat := 124159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events021.exact5540RawTerms
def rightRaw : List Term := Proof.Events467.exact119778RawTerms
def group : MergeGroup := .operator 5540 119778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5540) (leftOrdinal := 0)
    (rightResult := 119778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124159

namespace LeftMerge124164
def owner : Owner := ⟨.program ⟨257⟩, ⟨8143⟩⟩
def mergeEvent : Nat := 124164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }
def leftRaw : List Term := Proof.Events467.exact119648RawTerms
def rightRaw : List Term := Proof.Events084.exact21630RawTerms
def group : MergeGroup := .operator 119648 21630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119648) (leftOrdinal := 0)
    (rightResult := 21630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124164

namespace LeftMerge124181
def owner : Owner := ⟨.program ⟨257⟩, ⟨62364⟩⟩
def mergeEvent : Nat := 124181
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124175RawTerms
def rightRaw : List Term := Proof.Events084.exact21619RawTerms
def group : MergeGroup := .operator 124175 21619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124175) (leftOrdinal := 1)
    (rightResult := 21619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124181

namespace LeftMerge124183
def owner : Owner := ⟨.program ⟨257⟩, ⟨62364⟩⟩
def mergeEvent : Nat := 124183
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }
def rhsRaw : List Term := Proof.Events084.exact21589RawTerms
def group : MergeGroup := .relation 124182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124182) (rhsResult := 21589)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9538⟩⟩) ⟨7275⟩ 21589) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124183

namespace LeftMerge124184
def owner : Owner := ⟨.program ⟨257⟩, ⟨62364⟩⟩
def mergeEvent : Nat := 124184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124175RawTerms
def rightRaw : List Term := Proof.Events084.exact21619RawTerms
def group : MergeGroup := .operator 124175 21619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124175) (leftOrdinal := 0)
    (rightResult := 21619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124184

namespace LeftMerge124189
def owner : Owner := ⟨.program ⟨257⟩, ⟨62365⟩⟩
def mergeEvent : Nat := 124189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124185RawTerms
def rightRaw : List Term := Proof.Events484.exact124155RawTerms
def group : MergeGroup := .operator 124185 124155
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124185) (leftOrdinal := 1)
    (rightResult := 124155) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124189

namespace LeftMerge124197
def owner : Owner := ⟨.program ⟨257⟩, ⟨64396⟩⟩
def mergeEvent : Nat := 124197
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124191RawTerms
def rightRaw : List Term := Proof.Events484.exact124127RawTerms
def group : MergeGroup := .operator 124191 124127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124191) (leftOrdinal := 1)
    (rightResult := 124127) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124197

namespace LeftMerge124199
def owner : Owner := ⟨.program ⟨257⟩, ⟨64396⟩⟩
def mergeEvent : Nat := 124199
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63905⟩⟩] } }
def rhsRaw : List Term := Proof.Events484.exact124124RawTerms
def group : MergeGroup := .relation 124198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124198) (rhsResult := 124124)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64395⟩⟩) ⟨63905⟩ 124124) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63905⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨63905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124199

namespace LeftMerge124200
def owner : Owner := ⟨.program ⟨257⟩, ⟨64396⟩⟩
def mergeEvent : Nat := 124200
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124191RawTerms
def rightRaw : List Term := Proof.Events484.exact124127RawTerms
def group : MergeGroup := .operator 124191 124127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124191) (leftOrdinal := 0)
    (rightResult := 124127) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124200

namespace LeftMerge124214
def owner : Owner := ⟨.program ⟨257⟩, ⟨63332⟩⟩
def mergeEvent : Nat := 124214
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events485.exact124208RawTerms
def group : MergeGroup := .operator 119870 124208
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 124208) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124214

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
