import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge302095
def owner : Owner := ⟨.program ⟨257⟩, ⟨7425⟩⟩
def mergeEvent : Nat := 302095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events098.exact25137RawTerms
def group : MergeGroup := .operator 27 25137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 25137) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302095

namespace LeftMerge302112
def owner : Owner := ⟨.program ⟨257⟩, ⟨12535⟩⟩
def mergeEvent : Nat := 302112
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302106RawTerms
def rightRaw : List Term := Proof.Events098.exact25126RawTerms
def group : MergeGroup := .operator 302106 25126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302106) (leftOrdinal := 1)
    (rightResult := 25126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302112

namespace LeftMerge302114
def owner : Owner := ⟨.program ⟨257⟩, ⟨12535⟩⟩
def mergeEvent : Nat := 302114
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def rhsRaw : List Term := Proof.Events098.exact25096RawTerms
def group : MergeGroup := .relation 302113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302113) (rhsResult := 25096)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9571⟩⟩) ⟨7305⟩ 25096) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302114

namespace LeftMerge302115
def owner : Owner := ⟨.program ⟨257⟩, ⟨12535⟩⟩
def mergeEvent : Nat := 302115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302106RawTerms
def rightRaw : List Term := Proof.Events098.exact25126RawTerms
def group : MergeGroup := .operator 302106 25126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302106) (leftOrdinal := 0)
    (rightResult := 25126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302115

namespace LeftMerge302120
def owner : Owner := ⟨.program ⟨257⟩, ⟨18041⟩⟩
def mergeEvent : Nat := 302120
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302116RawTerms
def rightRaw : List Term := Proof.Events1180.exact302086RawTerms
def group : MergeGroup := .operator 302116 302086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302116) (leftOrdinal := 1)
    (rightResult := 302086) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302120

namespace LeftMerge302128
def owner : Owner := ⟨.program ⟨257⟩, ⟨20110⟩⟩
def mergeEvent : Nat := 302128
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302122RawTerms
def rightRaw : List Term := Proof.Events1179.exact302058RawTerms
def group : MergeGroup := .operator 302122 302058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302122) (leftOrdinal := 1)
    (rightResult := 302058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302128

namespace LeftMerge302130
def owner : Owner := ⟨.program ⟨257⟩, ⟨20110⟩⟩
def mergeEvent : Nat := 302130
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19649⟩⟩] } }
def rhsRaw : List Term := Proof.Events1179.exact302055RawTerms
def group : MergeGroup := .relation 302129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302129) (rhsResult := 302055)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20109⟩⟩) ⟨19649⟩ 302055) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19649⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], [⟨.program ⟨257⟩, ⟨19649⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302130

namespace LeftMerge302131
def owner : Owner := ⟨.program ⟨257⟩, ⟨20110⟩⟩
def mergeEvent : Nat := 302131
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302122RawTerms
def rightRaw : List Term := Proof.Events1179.exact302058RawTerms
def group : MergeGroup := .operator 302122 302058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302122) (leftOrdinal := 0)
    (rightResult := 302058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302131

namespace LeftMerge302145
def owner : Owner := ⟨.program ⟨257⟩, ⟨19052⟩⟩
def mergeEvent : Nat := 302145
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19049⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1180.exact302139RawTerms
def group : MergeGroup := .operator 295195 302139
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 302139) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19049⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19049⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302145

namespace LeftMerge302200
def owner : Owner := ⟨.program ⟨257⟩, ⟨18035⟩⟩
def mergeEvent : Nat := 302200
def frameStart : Nat := 302182
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1180.exact302196RawTerms
def rightRaw : List Term := Proof.Events1180.exact302193RawTerms
def group : MergeGroup := .operator 302196 302193
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302196) (leftOrdinal := 0)
    (rightResult := 302193) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12531⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302200

namespace LeftMerge302230
def owner : Owner := ⟨.program ⟨257⟩, ⟨19948⟩⟩
def mergeEvent : Nat := 302230
def frameStart : Nat := 302182
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302226RawTerms
def rightRaw : List Term := Proof.Events1180.exact302224RawTerms
def group : MergeGroup := .operator 302226 302224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302226) (leftOrdinal := 0)
    (rightResult := 302224) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302230

namespace LeftMerge302253
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def mergeEvent : Nat := 302253
def frameStart : Nat := 302182
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302249RawTerms
def rightRaw : List Term := Proof.Events1180.exact302246RawTerms
def group : MergeGroup := .operator 302249 302246
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302249) (leftOrdinal := 0)
    (rightResult := 302246) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302253

namespace LeftMerge302262
def owner : Owner := ⟨.program ⟨257⟩, ⟨20112⟩⟩
def mergeEvent : Nat := 302262
def frameStart : Nat := 302182
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302258RawTerms
def rightRaw : List Term := Proof.Events1180.exact302215RawTerms
def group : MergeGroup := .operator 302258 302215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302258) (leftOrdinal := 0)
    (rightResult := 302215) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20109⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302262

namespace LeftMerge302263
def owner : Owner := ⟨.program ⟨257⟩, ⟨20112⟩⟩
def mergeEvent : Nat := 302263
def frameStart : Nat := 302182
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302258RawTerms
def rightRaw : List Term := Proof.Events1180.exact302215RawTerms
def group : MergeGroup := .operator 302258 302215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302258) (leftOrdinal := 1)
    (rightResult := 302215) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302263

namespace LeftMerge302265
def owner : Owner := ⟨.program ⟨257⟩, ⟨20112⟩⟩
def mergeEvent : Nat := 302265
def frameStart : Nat := 302182
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19649⟩⟩] } }
def rhsRaw : List Term := Proof.Events1180.exact302212RawTerms
def group : MergeGroup := .relation 302264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302264) (rhsResult := 302212)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20109⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20109⟩⟩) ⟨19649⟩ 302212) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19649⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12531⟩⟩, ⟨.program ⟨257⟩, ⟨18034⟩⟩], [⟨.program ⟨257⟩, ⟨19649⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302265

namespace LeftMerge302273
def owner : Owner := ⟨.program ⟨257⟩, ⟨18510⟩⟩
def mergeEvent : Nat := 302273
def frameStart : Nat := 302182
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302226RawTerms
def rightRaw : List Term := Proof.Events1180.exact302269RawTerms
def group : MergeGroup := .operator 302226 302269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302226) (leftOrdinal := 0)
    (rightResult := 302269) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18508⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge302273

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
