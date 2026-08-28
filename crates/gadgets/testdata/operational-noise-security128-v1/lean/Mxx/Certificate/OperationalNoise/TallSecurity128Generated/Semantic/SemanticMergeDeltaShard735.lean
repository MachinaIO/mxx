import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge121204
def owner : Owner := ⟨.program ⟨257⟩, ⟨43459⟩⟩
def mergeEvent : Nat := 121204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }
def rhsRaw : List Term := Proof.Events473.exact121201RawTerms
def group : MergeGroup := .relation 121203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121203) (rhsResult := 121201)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩]⟩) (none) 121201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121204

namespace LeftMerge121205
def owner : Owner := ⟨.program ⟨257⟩, ⟨43459⟩⟩
def mergeEvent : Nat := 121205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩] } }
def rhsRaw : List Term := Proof.Events473.exact121201RawTerms
def group : MergeGroup := .relation 121203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121203) (rhsResult := 121201)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩]⟩) (none) 121201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121205

namespace LeftMerge121206
def owner : Owner := ⟨.program ⟨257⟩, ⟨43459⟩⟩
def mergeEvent : Nat := 121206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43905⟩⟩] } }
def rhsRaw : List Term := Proof.Events473.exact121201RawTerms
def group : MergeGroup := .relation 121203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121203) (rhsResult := 121201)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩]⟩) (none) 121201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43905⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43905⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121206

namespace LeftMerge121207
def owner : Owner := ⟨.program ⟨257⟩, ⟨43459⟩⟩
def mergeEvent : Nat := 121207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42947⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events473.exact121201RawTerms
def group : MergeGroup := .relation 121203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121203) (rhsResult := 121201)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩]⟩) (none) 121201) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42947⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42947⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121207

namespace LeftMerge121212
def owner : Owner := ⟨.program ⟨257⟩, ⟨44572⟩⟩
def mergeEvent : Nat := 121212
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121208RawTerms
def rightRaw : List Term := Proof.Events472.exact121030RawTerms
def group : MergeGroup := .operator 121208 121030
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121208) (leftOrdinal := 0)
    (rightResult := 121030) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121212

namespace LeftMerge121213
def owner : Owner := ⟨.program ⟨257⟩, ⟨44572⟩⟩
def mergeEvent : Nat := 121213
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43905⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121208RawTerms
def rightRaw : List Term := Proof.Events472.exact121030RawTerms
def group : MergeGroup := .operator 121208 121030
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121208) (leftOrdinal := 2)
    (rightResult := 121030) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43905⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43905⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121213

namespace LeftMerge121239
def owner : Owner := ⟨.program ⟨257⟩, ⟨39701⟩⟩
def mergeEvent : Nat := 121239
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events021.exact5399RawTerms
def rightRaw : List Term := Proof.Events467.exact119778RawTerms
def group : MergeGroup := .operator 5399 119778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5399) (leftOrdinal := 0)
    (rightResult := 119778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121239

namespace LeftMerge121244
def owner : Owner := ⟨.program ⟨257⟩, ⟨8132⟩⟩
def mergeEvent : Nat := 121244
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def leftRaw : List Term := Proof.Events467.exact119648RawTerms
def rightRaw : List Term := Proof.Events072.exact18583RawTerms
def group : MergeGroup := .operator 119648 18583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119648) (leftOrdinal := 0)
    (rightResult := 18583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121244

namespace LeftMerge121261
def owner : Owner := ⟨.program ⟨257⟩, ⟨39704⟩⟩
def mergeEvent : Nat := 121261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121255RawTerms
def rightRaw : List Term := Proof.Events021.exact5402RawTerms
def group : MergeGroup := .operator 121255 5402
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121255) (leftOrdinal := 1)
    (rightResult := 5402) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121261

namespace LeftMerge121262
def owner : Owner := ⟨.program ⟨257⟩, ⟨39704⟩⟩
def mergeEvent : Nat := 121262
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121255RawTerms
def rightRaw : List Term := Proof.Events021.exact5402RawTerms
def group : MergeGroup := .operator 121255 5402
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121255) (leftOrdinal := 0)
    (rightResult := 5402) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121262

namespace LeftMerge121267
def owner : Owner := ⟨.program ⟨257⟩, ⟨14122⟩⟩
def mergeEvent : Nat := 121267
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events021.exact5402RawTerms
def rightRaw : List Term := Proof.Events467.exact119778RawTerms
def group : MergeGroup := .operator 5402 119778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5402) (leftOrdinal := 0)
    (rightResult := 119778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121267

namespace LeftMerge121272
def owner : Owner := ⟨.program ⟨257⟩, ⟨8149⟩⟩
def mergeEvent : Nat := 121272
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }
def leftRaw : List Term := Proof.Events467.exact119648RawTerms
def rightRaw : List Term := Proof.Events072.exact18624RawTerms
def group : MergeGroup := .operator 119648 18624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119648) (leftOrdinal := 0)
    (rightResult := 18624) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121272

namespace LeftMerge121289
def owner : Owner := ⟨.program ⟨257⟩, ⟨14125⟩⟩
def mergeEvent : Nat := 121289
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121283RawTerms
def rightRaw : List Term := Proof.Events072.exact18613RawTerms
def group : MergeGroup := .operator 121283 18613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121283) (leftOrdinal := 1)
    (rightResult := 18613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121289

namespace LeftMerge121291
def owner : Owner := ⟨.program ⟨257⟩, ⟨14125⟩⟩
def mergeEvent : Nat := 121291
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def rhsRaw : List Term := Proof.Events072.exact18583RawTerms
def group : MergeGroup := .relation 121290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121290) (rhsResult := 18583)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9556⟩⟩) ⟨7282⟩ 18583) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121291

namespace LeftMerge121292
def owner : Owner := ⟨.program ⟨257⟩, ⟨14125⟩⟩
def mergeEvent : Nat := 121292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121283RawTerms
def rightRaw : List Term := Proof.Events072.exact18613RawTerms
def group : MergeGroup := .operator 121283 18613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121283) (leftOrdinal := 0)
    (rightResult := 18613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121292

namespace LeftMerge121297
def owner : Owner := ⟨.program ⟨257⟩, ⟨39705⟩⟩
def mergeEvent : Nat := 121297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121293RawTerms
def rightRaw : List Term := Proof.Events473.exact121263RawTerms
def group : MergeGroup := .operator 121293 121263
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121293) (leftOrdinal := 1)
    (rightResult := 121263) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14121⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121297

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
