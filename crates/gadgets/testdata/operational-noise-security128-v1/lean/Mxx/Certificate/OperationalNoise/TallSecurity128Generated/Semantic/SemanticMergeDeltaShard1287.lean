import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge209058
def owner : Owner := ⟨.program ⟨257⟩, ⟨41620⟩⟩
def mergeEvent : Nat := 209058
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }
def leftRaw : List Term := Proof.Events816.exact209049RawTerms
def rightRaw : List Term := Proof.Events816.exact208985RawTerms
def group : MergeGroup := .operator 209049 208985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209049) (leftOrdinal := 0)
    (rightResult := 208985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41619⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209058

namespace LeftMerge209072
def owner : Owner := ⟨.program ⟨257⟩, ⟨40552⟩⟩
def mergeEvent : Nat := 209072
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events816.exact209066RawTerms
def group : MergeGroup := .operator 207620 209066
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 209066) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40549⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209072

namespace LeftMerge209151
def owner : Owner := ⟨.program ⟨257⟩, ⟨39795⟩⟩
def mergeEvent : Nat := 209151
def frameStart : Nat := 209121
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events816.exact209147RawTerms
def rightRaw : List Term := Proof.Events816.exact209144RawTerms
def group : MergeGroup := .operator 209147 209144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209147) (leftOrdinal := 0)
    (rightResult := 209144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209151

namespace LeftMerge209181
def owner : Owner := ⟨.program ⟨257⟩, ⟨41388⟩⟩
def mergeEvent : Nat := 209181
def frameStart : Nat := 209121
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209177RawTerms
def rightRaw : List Term := Proof.Events817.exact209175RawTerms
def group : MergeGroup := .operator 209177 209175
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209177) (leftOrdinal := 0)
    (rightResult := 209175) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209181

namespace LeftMerge209204
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 209204
def frameStart : Nat := 209121
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209200RawTerms
def rightRaw : List Term := Proof.Events817.exact209197RawTerms
def group : MergeGroup := .operator 209200 209197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209200) (leftOrdinal := 0)
    (rightResult := 209197) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209204

namespace LeftMerge209213
def owner : Owner := ⟨.program ⟨257⟩, ⟨41622⟩⟩
def mergeEvent : Nat := 209213
def frameStart : Nat := 209121
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209209RawTerms
def rightRaw : List Term := Proof.Events817.exact209166RawTerms
def group : MergeGroup := .operator 209209 209166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209209) (leftOrdinal := 0)
    (rightResult := 209166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41619⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209213

namespace LeftMerge209214
def owner : Owner := ⟨.program ⟨257⟩, ⟨41622⟩⟩
def mergeEvent : Nat := 209214
def frameStart : Nat := 209121
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209209RawTerms
def rightRaw : List Term := Proof.Events817.exact209166RawTerms
def group : MergeGroup := .operator 209209 209166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209209) (leftOrdinal := 1)
    (rightResult := 209166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41619⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209214

namespace LeftMerge209216
def owner : Owner := ⟨.program ⟨257⟩, ⟨41622⟩⟩
def mergeEvent : Nat := 209216
def frameStart : Nat := 209121
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41109⟩⟩] } }
def rhsRaw : List Term := Proof.Events817.exact209163RawTerms
def group : MergeGroup := .relation 209215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209215) (rhsResult := 209163)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41619⟩⟩) ⟨41109⟩ 209163) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41109⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], [⟨.program ⟨257⟩, ⟨41109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209216

namespace LeftMerge209224
def owner : Owner := ⟨.program ⟨257⟩, ⟨40110⟩⟩
def mergeEvent : Nat := 209224
def frameStart : Nat := 209121
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209177RawTerms
def rightRaw : List Term := Proof.Events817.exact209220RawTerms
def group : MergeGroup := .operator 209177 209220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209177) (leftOrdinal := 0)
    (rightResult := 209220) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209224

namespace LeftMerge209241
def owner : Owner := ⟨.program ⟨257⟩, ⟨40552⟩⟩
def mergeEvent : Nat := 209241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events817.exact209238RawTerms
def group : MergeGroup := .relation 209240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209240) (rhsResult := 209238)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209239 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩]⟩) (none) 209238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209241

namespace LeftMerge209242
def owner : Owner := ⟨.program ⟨257⟩, ⟨40552⟩⟩
def mergeEvent : Nat := 209242
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }
def rhsRaw : List Term := Proof.Events817.exact209238RawTerms
def group : MergeGroup := .relation 209240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209240) (rhsResult := 209238)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209239 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩]⟩) (none) 209238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209242

namespace LeftMerge209243
def owner : Owner := ⟨.program ⟨257⟩, ⟨40552⟩⟩
def mergeEvent : Nat := 209243
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41109⟩⟩] } }
def rhsRaw : List Term := Proof.Events817.exact209238RawTerms
def group : MergeGroup := .relation 209240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209240) (rhsResult := 209238)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209239 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩]⟩) (none) 209238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41109⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], [⟨.program ⟨257⟩, ⟨41109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209243

namespace LeftMerge209244
def owner : Owner := ⟨.program ⟨257⟩, ⟨40552⟩⟩
def mergeEvent : Nat := 209244
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events817.exact209238RawTerms
def group : MergeGroup := .relation 209240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209240) (rhsResult := 209238)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209239 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40549⟩⟩]⟩) (none) 209238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209244

namespace LeftMerge209249
def owner : Owner := ⟨.program ⟨257⟩, ⟨41621⟩⟩
def mergeEvent : Nat := 209249
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41109⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209245RawTerms
def rightRaw : List Term := Proof.Events816.exact209059RawTerms
def group : MergeGroup := .operator 209245 209059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209245) (leftOrdinal := 2)
    (rightResult := 209059) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41109⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], [⟨.program ⟨257⟩, ⟨41109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209249

namespace LeftMerge209250
def owner : Owner := ⟨.program ⟨257⟩, ⟨41621⟩⟩
def mergeEvent : Nat := 209250
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209245RawTerms
def rightRaw : List Term := Proof.Events816.exact209059RawTerms
def group : MergeGroup := .operator 209245 209059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209245) (leftOrdinal := 1)
    (rightResult := 209059) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209250

namespace LeftMerge209258
def owner : Owner := ⟨.program ⟨257⟩, ⟨41991⟩⟩
def mergeEvent : Nat := 209258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }
def leftRaw : List Term := Proof.Events817.exact209252RawTerms
def rightRaw : List Term := Proof.Events816.exact208975RawTerms
def group : MergeGroup := .operator 209252 208975
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209252) (leftOrdinal := 0)
    (rightResult := 208975) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209258

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
