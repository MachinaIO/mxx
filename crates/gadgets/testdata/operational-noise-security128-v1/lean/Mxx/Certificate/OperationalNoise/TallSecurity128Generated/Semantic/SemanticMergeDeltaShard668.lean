import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge111035
def owner : Owner := ⟨.program ⟨257⟩, ⟨54442⟩⟩
def mergeEvent : Nat := 111035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events433.exact111029RawTerms
def group : MergeGroup := .operator 105245 111029
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 111029) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54439⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111035

namespace LeftMerge111114
def owner : Owner := ⟨.program ⟨257⟩, ⟨53553⟩⟩
def mergeEvent : Nat := 111114
def frameStart : Nat := 111084
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events434.exact111110RawTerms
def rightRaw : List Term := Proof.Events434.exact111107RawTerms
def group : MergeGroup := .operator 111110 111107
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111110) (leftOrdinal := 0)
    (rightResult := 111107) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24782⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111114

namespace LeftMerge111144
def owner : Owner := ⟨.program ⟨257⟩, ⟨55272⟩⟩
def mergeEvent : Nat := 111144
def frameStart : Nat := 111084
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events434.exact111140RawTerms
def rightRaw : List Term := Proof.Events434.exact111138RawTerms
def group : MergeGroup := .operator 111140 111138
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111140) (leftOrdinal := 0)
    (rightResult := 111138) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111144

namespace LeftMerge111167
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 111167
def frameStart : Nat := 111084
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events434.exact111163RawTerms
def rightRaw : List Term := Proof.Events434.exact111160RawTerms
def group : MergeGroup := .operator 111163 111160
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111163) (leftOrdinal := 0)
    (rightResult := 111160) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111167

namespace LeftMerge111176
def owner : Owner := ⟨.program ⟨257⟩, ⟨55513⟩⟩
def mergeEvent : Nat := 111176
def frameStart : Nat := 111084
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩] } }
def leftRaw : List Term := Proof.Events434.exact111172RawTerms
def rightRaw : List Term := Proof.Events434.exact111129RawTerms
def group : MergeGroup := .operator 111172 111129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111172) (leftOrdinal := 0)
    (rightResult := 111129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55510⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111176

namespace LeftMerge111177
def owner : Owner := ⟨.program ⟨257⟩, ⟨55513⟩⟩
def mergeEvent : Nat := 111177
def frameStart : Nat := 111084
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩] } }
def leftRaw : List Term := Proof.Events434.exact111172RawTerms
def rightRaw : List Term := Proof.Events434.exact111129RawTerms
def group : MergeGroup := .operator 111172 111129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111172) (leftOrdinal := 1)
    (rightResult := 111129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55510⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111177

namespace LeftMerge111179
def owner : Owner := ⟨.program ⟨257⟩, ⟨55513⟩⟩
def mergeEvent : Nat := 111179
def frameStart : Nat := 111084
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54995⟩⟩] } }
def rhsRaw : List Term := Proof.Events434.exact111126RawTerms
def group : MergeGroup := .relation 111178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111178) (rhsResult := 111126)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55510⟩⟩) ⟨54995⟩ 111126) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54995⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], [⟨.program ⟨257⟩, ⟨54995⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111179

namespace LeftMerge111187
def owner : Owner := ⟨.program ⟨257⟩, ⟨53878⟩⟩
def mergeEvent : Nat := 111187
def frameStart : Nat := 111084
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events434.exact111140RawTerms
def rightRaw : List Term := Proof.Events434.exact111183RawTerms
def group : MergeGroup := .operator 111140 111183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111140) (leftOrdinal := 0)
    (rightResult := 111183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53876⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111187

namespace LeftMerge111204
def owner : Owner := ⟨.program ⟨257⟩, ⟨54442⟩⟩
def mergeEvent : Nat := 111204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events434.exact111201RawTerms
def group : MergeGroup := .relation 111203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111203) (rhsResult := 111201)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 111202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩]⟩) (none) 111201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111204

namespace LeftMerge111205
def owner : Owner := ⟨.program ⟨257⟩, ⟨54442⟩⟩
def mergeEvent : Nat := 111205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩] } }
def rhsRaw : List Term := Proof.Events434.exact111201RawTerms
def group : MergeGroup := .relation 111203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111203) (rhsResult := 111201)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 111202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩]⟩) (none) 111201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111205

namespace LeftMerge111206
def owner : Owner := ⟨.program ⟨257⟩, ⟨54442⟩⟩
def mergeEvent : Nat := 111206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54995⟩⟩] } }
def rhsRaw : List Term := Proof.Events434.exact111201RawTerms
def group : MergeGroup := .relation 111203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111203) (rhsResult := 111201)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 111202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩]⟩) (none) 111201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54995⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], [⟨.program ⟨257⟩, ⟨54995⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111206

namespace LeftMerge111207
def owner : Owner := ⟨.program ⟨257⟩, ⟨54442⟩⟩
def mergeEvent : Nat := 111207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨53876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events434.exact111201RawTerms
def group : MergeGroup := .relation 111203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 111203) (rhsResult := 111201)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 111202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54439⟩⟩]⟩) (none) 111201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨53876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111207

namespace LeftMerge111212
def owner : Owner := ⟨.program ⟨257⟩, ⟨55512⟩⟩
def mergeEvent : Nat := 111212
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54995⟩⟩] } }
def leftRaw : List Term := Proof.Events434.exact111208RawTerms
def rightRaw : List Term := Proof.Events433.exact111022RawTerms
def group : MergeGroup := .operator 111208 111022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111208) (leftOrdinal := 2)
    (rightResult := 111022) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54995⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54995⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], [⟨.program ⟨257⟩, ⟨54995⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111212

namespace LeftMerge111213
def owner : Owner := ⟨.program ⟨257⟩, ⟨55512⟩⟩
def mergeEvent : Nat := 111213
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩] } }
def leftRaw : List Term := Proof.Events434.exact111208RawTerms
def rightRaw : List Term := Proof.Events433.exact111022RawTerms
def group : MergeGroup := .operator 111208 111022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111208) (leftOrdinal := 1)
    (rightResult := 111022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55510⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111213

namespace LeftMerge111221
def owner : Owner := ⟨.program ⟨257⟩, ⟨55965⟩⟩
def mergeEvent : Nat := 111221
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55963⟩⟩] } }
def leftRaw : List Term := Proof.Events434.exact111215RawTerms
def rightRaw : List Term := Proof.Events433.exact110938RawTerms
def group : MergeGroup := .operator 111215 110938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111215) (leftOrdinal := 0)
    (rightResult := 110938) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55963⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55963⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge111221

namespace LeftMerge111222
def owner : Owner := ⟨.program ⟨257⟩, ⟨55965⟩⟩
def mergeEvent : Nat := 111222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨53876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55963⟩⟩] } }
def leftRaw : List Term := Proof.Events434.exact111215RawTerms
def rightRaw : List Term := Proof.Events433.exact110938RawTerms
def group : MergeGroup := .operator 111215 110938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 111215) (leftOrdinal := 1)
    (rightResult := 110938) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨53876⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55963⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨53876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55963⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge111222

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
