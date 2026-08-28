import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge73055
def owner : Owner := ⟨.program ⟨214⟩, ⟨7200⟩⟩
def mergeEvent : Nat := 73055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65165RawTerms
def rightRaw : List Term := Proof.Events056.exact14529RawTerms
def group : MergeGroup := .operator 65165 14529
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65165) (leftOrdinal := 0)
    (rightResult := 14529) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73055

namespace LeftMerge73072
def owner : Owner := ⟨.program ⟨214⟩, ⟨9504⟩⟩
def mergeEvent : Nat := 73072
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact73066RawTerms
def rightRaw : List Term := Proof.Events056.exact14518RawTerms
def group : MergeGroup := .operator 73066 14518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73066) (leftOrdinal := 1)
    (rightResult := 14518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7834⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73072

namespace LeftMerge73074
def owner : Owner := ⟨.program ⟨214⟩, ⟨9504⟩⟩
def mergeEvent : Nat := 73074
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }
def rhsRaw : List Term := Proof.Events056.exact14488RawTerms
def group : MergeGroup := .relation 73073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73073) (rhsResult := 14488)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7834⟩⟩) ⟨6773⟩ 14488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73074

namespace LeftMerge73075
def owner : Owner := ⟨.program ⟨214⟩, ⟨9504⟩⟩
def mergeEvent : Nat := 73075
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact73066RawTerms
def rightRaw : List Term := Proof.Events056.exact14518RawTerms
def group : MergeGroup := .operator 73066 14518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73066) (leftOrdinal := 0)
    (rightResult := 14518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7834⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73075

namespace LeftMerge73080
def owner : Owner := ⟨.program ⟨214⟩, ⟨10675⟩⟩
def mergeEvent : Nat := 73080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact73076RawTerms
def rightRaw : List Term := Proof.Events285.exact73046RawTerms
def group : MergeGroup := .operator 73076 73046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73076) (leftOrdinal := 1)
    (rightResult := 73046) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73080

namespace LeftMerge73088
def owner : Owner := ⟨.program ⟨214⟩, ⟨24984⟩⟩
def mergeEvent : Nat := 73088
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact73082RawTerms
def rightRaw : List Term := Proof.Events285.exact73018RawTerms
def group : MergeGroup := .operator 73082 73018
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73082) (leftOrdinal := 1)
    (rightResult := 73018) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24983⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73088

namespace LeftMerge73090
def owner : Owner := ⟨.program ⟨214⟩, ⟨24984⟩⟩
def mergeEvent : Nat := 73090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }
def rhsRaw : List Term := Proof.Events285.exact73015RawTerms
def group : MergeGroup := .relation 73089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73089) (rhsResult := 73015)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24983⟩⟩) ⟨22994⟩ 73015) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨22994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73090

namespace LeftMerge73091
def owner : Owner := ⟨.program ⟨214⟩, ⟨24984⟩⟩
def mergeEvent : Nat := 73091
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact73082RawTerms
def rightRaw : List Term := Proof.Events285.exact73018RawTerms
def group : MergeGroup := .operator 73082 73018
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73082) (leftOrdinal := 0)
    (rightResult := 73018) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24983⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73091

namespace LeftMerge73105
def owner : Owner := ⟨.program ⟨214⟩, ⟨19095⟩⟩
def mergeEvent : Nat := 73105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events285.exact73099RawTerms
def group : MergeGroup := .operator 65387 73099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 73099) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19092⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73105

namespace LeftMerge73184
def owner : Owner := ⟨.program ⟨214⟩, ⟨10669⟩⟩
def mergeEvent : Nat := 73184
def frameStart : Nat := 73154
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events285.exact73180RawTerms
def rightRaw : List Term := Proof.Events285.exact73177RawTerms
def group : MergeGroup := .operator 73180 73177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73180) (leftOrdinal := 0)
    (rightResult := 73177) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73184

namespace LeftMerge73214
def owner : Owner := ⟨.program ⟨214⟩, ⟨10770⟩⟩
def mergeEvent : Nat := 73214
def frameStart : Nat := 73154
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact73210RawTerms
def rightRaw : List Term := Proof.Events285.exact73208RawTerms
def group : MergeGroup := .operator 73210 73208
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73210) (leftOrdinal := 0)
    (rightResult := 73208) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73214

namespace LeftMerge73237
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def mergeEvent : Nat := 73237
def frameStart : Nat := 73154
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73233RawTerms
def rightRaw : List Term := Proof.Events286.exact73230RawTerms
def group : MergeGroup := .operator 73233 73230
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73233) (leftOrdinal := 0)
    (rightResult := 73230) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7834⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73237

namespace LeftMerge73246
def owner : Owner := ⟨.program ⟨214⟩, ⟨24986⟩⟩
def mergeEvent : Nat := 73246
def frameStart : Nat := 73154
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73242RawTerms
def rightRaw : List Term := Proof.Events285.exact73199RawTerms
def group : MergeGroup := .operator 73242 73199
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73242) (leftOrdinal := 0)
    (rightResult := 73199) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24983⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73246

namespace LeftMerge73247
def owner : Owner := ⟨.program ⟨214⟩, ⟨24986⟩⟩
def mergeEvent : Nat := 73247
def frameStart : Nat := 73154
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73242RawTerms
def rightRaw : List Term := Proof.Events285.exact73199RawTerms
def group : MergeGroup := .operator 73242 73199
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73242) (leftOrdinal := 1)
    (rightResult := 73199) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24983⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73247

namespace LeftMerge73249
def owner : Owner := ⟨.program ⟨214⟩, ⟨24986⟩⟩
def mergeEvent : Nat := 73249
def frameStart : Nat := 73154
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }
def rhsRaw : List Term := Proof.Events285.exact73196RawTerms
def group : MergeGroup := .relation 73248
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73248) (rhsResult := 73196)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24983⟩⟩) ⟨22994⟩ 73196) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨22994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73249

namespace LeftMerge73257
def owner : Owner := ⟨.program ⟨214⟩, ⟨14951⟩⟩
def mergeEvent : Nat := 73257
def frameStart : Nat := 73154
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events285.exact73210RawTerms
def rightRaw : List Term := Proof.Events286.exact73253RawTerms
def group : MergeGroup := .operator 73210 73253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73210) (leftOrdinal := 0)
    (rightResult := 73253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73257

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
