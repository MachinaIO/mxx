import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge73274
def owner : Owner := ⟨.program ⟨214⟩, ⟨19095⟩⟩
def mergeEvent : Nat := 73274
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }
def rhsRaw : List Term := Proof.Events286.exact73271RawTerms
def group : MergeGroup := .relation 73273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73273) (rhsResult := 73271)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73272 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩) (none) 73271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73274

namespace LeftMerge73275
def owner : Owner := ⟨.program ⟨214⟩, ⟨19095⟩⟩
def mergeEvent : Nat := 73275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } }
def rhsRaw : List Term := Proof.Events286.exact73271RawTerms
def group : MergeGroup := .relation 73273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73273) (rhsResult := 73271)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73272 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩) (none) 73271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73275

namespace LeftMerge73276
def owner : Owner := ⟨.program ⟨214⟩, ⟨19095⟩⟩
def mergeEvent : Nat := 73276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }
def rhsRaw : List Term := Proof.Events286.exact73271RawTerms
def group : MergeGroup := .relation 73273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73273) (rhsResult := 73271)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73272 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩) (none) 73271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨22994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73276

namespace LeftMerge73277
def owner : Owner := ⟨.program ⟨214⟩, ⟨19095⟩⟩
def mergeEvent : Nat := 73277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events286.exact73271RawTerms
def group : MergeGroup := .relation 73273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73273) (rhsResult := 73271)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73272 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19092⟩⟩]⟩) (none) 73271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73277

namespace LeftMerge73282
def owner : Owner := ⟨.program ⟨214⟩, ⟨24985⟩⟩
def mergeEvent : Nat := 73282
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73278RawTerms
def rightRaw : List Term := Proof.Events285.exact73092RawTerms
def group : MergeGroup := .operator 73278 73092
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73278) (leftOrdinal := 2)
    (rightResult := 73092) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨22994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73282

namespace LeftMerge73283
def owner : Owner := ⟨.program ⟨214⟩, ⟨24985⟩⟩
def mergeEvent : Nat := 73283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73278RawTerms
def rightRaw : List Term := Proof.Events285.exact73092RawTerms
def group : MergeGroup := .operator 73278 73092
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73278) (leftOrdinal := 1)
    (rightResult := 73092) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73283

namespace LeftMerge73291
def owner : Owner := ⟨.program ⟨214⟩, ⟨26553⟩⟩
def mergeEvent : Nat := 73291
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73285RawTerms
def rightRaw : List Term := Proof.Events285.exact73008RawTerms
def group : MergeGroup := .operator 73285 73008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73285) (leftOrdinal := 0)
    (rightResult := 73008) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26551⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73291

namespace LeftMerge73292
def owner : Owner := ⟨.program ⟨214⟩, ⟨26553⟩⟩
def mergeEvent : Nat := 73292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73285RawTerms
def rightRaw : List Term := Proof.Events285.exact73008RawTerms
def group : MergeGroup := .operator 73285 73008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73285) (leftOrdinal := 1)
    (rightResult := 73008) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26551⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73292

namespace LeftMerge73294
def owner : Owner := ⟨.program ⟨214⟩, ⟨26553⟩⟩
def mergeEvent : Nat := 73294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23781⟩⟩] } }
def rhsRaw : List Term := Proof.Events285.exact73005RawTerms
def group : MergeGroup := .relation 73293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73293) (rhsResult := 73005)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26551⟩⟩) ⟨23781⟩ 73005) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23781⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23781⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73294

namespace LeftMerge73308
def owner : Owner := ⟨.program ⟨214⟩, ⟨20535⟩⟩
def mergeEvent : Nat := 73308
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20532⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events286.exact73302RawTerms
def group : MergeGroup := .operator 65387 73302
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 73302) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20532⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73308

namespace LeftMerge73429
def owner : Owner := ⟨.program ⟨214⟩, ⟨14991⟩⟩
def mergeEvent : Nat := 73429
def frameStart : Nat := 73363
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73425RawTerms
def rightRaw : List Term := Proof.Events286.exact73423RawTerms
def group : MergeGroup := .operator 73425 73423
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73425) (leftOrdinal := 0)
    (rightResult := 73423) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73429

namespace LeftMerge73441
def owner : Owner := ⟨.program ⟨214⟩, ⟨26552⟩⟩
def mergeEvent : Nat := 73441
def frameStart : Nat := 73363
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73437RawTerms
def rightRaw : List Term := Proof.Events286.exact73414RawTerms
def group : MergeGroup := .operator 73437 73414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73437) (leftOrdinal := 0)
    (rightResult := 73414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26551⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73441

namespace LeftMerge73442
def owner : Owner := ⟨.program ⟨214⟩, ⟨26552⟩⟩
def mergeEvent : Nat := 73442
def frameStart : Nat := 73363
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73437RawTerms
def rightRaw : List Term := Proof.Events286.exact73414RawTerms
def group : MergeGroup := .operator 73437 73414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73437) (leftOrdinal := 1)
    (rightResult := 73414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26551⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73442

namespace LeftMerge73444
def owner : Owner := ⟨.program ⟨214⟩, ⟨26552⟩⟩
def mergeEvent : Nat := 73444
def frameStart : Nat := 73363
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23781⟩⟩] } }
def rhsRaw : List Term := Proof.Events286.exact73411RawTerms
def group : MergeGroup := .relation 73443
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73443) (rhsResult := 73411)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26551⟩⟩) ⟨23781⟩ 73411) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23781⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23781⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge73444

namespace LeftMerge73452
def owner : Owner := ⟨.program ⟨214⟩, ⟨15308⟩⟩
def mergeEvent : Nat := 73452
def frameStart : Nat := 73363
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events286.exact73425RawTerms
def rightRaw : List Term := Proof.Events286.exact73448RawTerms
def group : MergeGroup := .operator 73425 73448
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 73425) (leftOrdinal := 0)
    (rightResult := 73448) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73452

namespace LeftMerge73469
def owner : Owner := ⟨.program ⟨214⟩, ⟨20535⟩⟩
def mergeEvent : Nat := 73469
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }
def rhsRaw : List Term := Proof.Events286.exact73466RawTerms
def group : MergeGroup := .relation 73468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 73468) (rhsResult := 73466)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20532⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 73467 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20532⟩⟩]⟩) (none) 73466) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge73469

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
