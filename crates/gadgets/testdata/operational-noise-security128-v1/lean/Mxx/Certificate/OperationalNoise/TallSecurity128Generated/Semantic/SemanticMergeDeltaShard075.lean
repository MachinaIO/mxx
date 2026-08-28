import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge16161
def owner : Owner := ⟨.program ⟨257⟩, ⟨9669⟩⟩
def mergeEvent : Nat := 16161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩, ⟨.program ⟨257⟩, ⟨7101⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16157RawTerms
def rightRaw : List Term := Proof.Events063.exact16134RawTerms
def group : MergeGroup := .operator 16157 16134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16157) (leftOrdinal := 0)
    (rightResult := 16134) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7101⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩, ⟨.program ⟨257⟩, ⟨7101⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16161

namespace LeftMerge16166
def owner : Owner := ⟨.program ⟨257⟩, ⟨7027⟩⟩
def mergeEvent : Nat := 16166
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 2 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16166

namespace LeftMerge16191
def owner : Owner := ⟨.program ⟨257⟩, ⟨9590⟩⟩
def mergeEvent : Nat := 16191
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16187RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16187 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16187) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7244⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16191

namespace LeftMerge16196
def owner : Owner := ⟨.program ⟨257⟩, ⟨9651⟩⟩
def mergeEvent : Nat := 16196
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16192RawTerms
def rightRaw : List Term := Proof.Events063.exact16184RawTerms
def group : MergeGroup := .operator 16192 16184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16192) (leftOrdinal := 0)
    (rightResult := 16184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9501⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16196

namespace LeftMerge16201
def owner : Owner := ⟨.program ⟨257⟩, ⟨9670⟩⟩
def mergeEvent : Nat := 16201
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩, ⟨.program ⟨257⟩, ⟨7123⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16197RawTerms
def rightRaw : List Term := Proof.Events063.exact16174RawTerms
def group : MergeGroup := .operator 16197 16174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16197) (leftOrdinal := 0)
    (rightResult := 16174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7123⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩, ⟨.program ⟨257⟩, ⟨7123⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16201

namespace LeftMerge16206
def owner : Owner := ⟨.program ⟨257⟩, ⟨7025⟩⟩
def mergeEvent : Nat := 16206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events017.exact4569RawTerms
def group : MergeGroup := .operator 2 4569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 4569) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6753⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6753⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16206

namespace LeftMerge16231
def owner : Owner := ⟨.program ⟨257⟩, ⟨9591⟩⟩
def mergeEvent : Nat := 16231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7246⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16227RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16227 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16227) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7246⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7246⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16231

namespace LeftMerge16236
def owner : Owner := ⟨.program ⟨257⟩, ⟨9652⟩⟩
def mergeEvent : Nat := 16236
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7246⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9503⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16232RawTerms
def rightRaw : List Term := Proof.Events063.exact16224RawTerms
def group : MergeGroup := .operator 16232 16224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16232) (leftOrdinal := 0)
    (rightResult := 16224) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7246⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9503⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7246⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9503⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16236

namespace LeftMerge16241
def owner : Owner := ⟨.program ⟨257⟩, ⟨9671⟩⟩
def mergeEvent : Nat := 16241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7246⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9503⟩⟩, ⟨.program ⟨257⟩, ⟨7119⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16237RawTerms
def rightRaw : List Term := Proof.Events063.exact16214RawTerms
def group : MergeGroup := .operator 16237 16214
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16237) (leftOrdinal := 0)
    (rightResult := 16214) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7246⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9503⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7119⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7246⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9503⟩⟩, ⟨.program ⟨257⟩, ⟨7119⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16241

namespace LeftMerge16246
def owner : Owner := ⟨.program ⟨257⟩, ⟨7021⟩⟩
def mergeEvent : Nat := 16246
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events020.exact5317RawTerms
def group : MergeGroup := .operator 2 5317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 5317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16246

namespace LeftMerge16271
def owner : Owner := ⟨.program ⟨257⟩, ⟨9592⟩⟩
def mergeEvent : Nat := 16271
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16267RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16267 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16267) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7248⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16271

namespace LeftMerge16276
def owner : Owner := ⟨.program ⟨257⟩, ⟨9653⟩⟩
def mergeEvent : Nat := 16276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9505⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16272RawTerms
def rightRaw : List Term := Proof.Events063.exact16264RawTerms
def group : MergeGroup := .operator 16272 16264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16272) (leftOrdinal := 0)
    (rightResult := 16264) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9505⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9505⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16276

namespace LeftMerge16281
def owner : Owner := ⟨.program ⟨257⟩, ⟨9672⟩⟩
def mergeEvent : Nat := 16281
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9505⟩⟩, ⟨.program ⟨257⟩, ⟨7111⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16277RawTerms
def rightRaw : List Term := Proof.Events063.exact16254RawTerms
def group : MergeGroup := .operator 16277 16254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16277) (leftOrdinal := 0)
    (rightResult := 16254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9505⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7111⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9505⟩⟩, ⟨.program ⟨257⟩, ⟨7111⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16281

namespace LeftMerge16286
def owner : Owner := ⟨.program ⟨257⟩, ⟨7024⟩⟩
def mergeEvent : Nat := 16286
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events023.exact6065RawTerms
def group : MergeGroup := .operator 2 6065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 6065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16286

namespace LeftMerge16311
def owner : Owner := ⟨.program ⟨257⟩, ⟨9593⟩⟩
def mergeEvent : Nat := 16311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16307RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16307 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16307) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7250⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16311

namespace LeftMerge16316
def owner : Owner := ⟨.program ⟨257⟩, ⟨9654⟩⟩
def mergeEvent : Nat := 16316
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9507⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16312RawTerms
def rightRaw : List Term := Proof.Events063.exact16304RawTerms
def group : MergeGroup := .operator 16312 16304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16312) (leftOrdinal := 0)
    (rightResult := 16304) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9507⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9507⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16316

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
