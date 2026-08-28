import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge95444
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def mergeEvent : Nat := 95444
def frameStart : Nat := 95373
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events372.exact95440RawTerms
def rightRaw : List Term := Proof.Events372.exact95437RawTerms
def group : MergeGroup := .operator 95440 95437
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95440) (leftOrdinal := 0)
    (rightResult := 95437) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95444

namespace LeftMerge95453
def owner : Owner := ⟨.program ⟨214⟩, ⟨25594⟩⟩
def mergeEvent : Nat := 95453
def frameStart : Nat := 95373
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩] } }
def leftRaw : List Term := Proof.Events372.exact95449RawTerms
def rightRaw : List Term := Proof.Events372.exact95406RawTerms
def group : MergeGroup := .operator 95449 95406
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95449) (leftOrdinal := 0)
    (rightResult := 95406) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25591⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95453

namespace LeftMerge95454
def owner : Owner := ⟨.program ⟨214⟩, ⟨25594⟩⟩
def mergeEvent : Nat := 95454
def frameStart : Nat := 95373
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩] } }
def leftRaw : List Term := Proof.Events372.exact95449RawTerms
def rightRaw : List Term := Proof.Events372.exact95406RawTerms
def group : MergeGroup := .operator 95449 95406
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95449) (leftOrdinal := 1)
    (rightResult := 95406) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25591⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95454

namespace LeftMerge95456
def owner : Owner := ⟨.program ⟨214⟩, ⟨25594⟩⟩
def mergeEvent : Nat := 95456
def frameStart : Nat := 95373
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23326⟩⟩] } }
def rhsRaw : List Term := Proof.Events372.exact95403RawTerms
def group : MergeGroup := .relation 95455
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95455) (rhsResult := 95403)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25591⟩⟩) ⟨23326⟩ 95403) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23326⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], [⟨.program ⟨214⟩, ⟨23326⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95456

namespace LeftMerge95464
def owner : Owner := ⟨.program ⟨214⟩, ⟨16744⟩⟩
def mergeEvent : Nat := 95464
def frameStart : Nat := 95373
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events372.exact95417RawTerms
def rightRaw : List Term := Proof.Events372.exact95460RawTerms
def group : MergeGroup := .operator 95417 95460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95417) (leftOrdinal := 0)
    (rightResult := 95460) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95464

namespace LeftMerge95481
def owner : Owner := ⟨.program ⟨214⟩, ⟨20096⟩⟩
def mergeEvent : Nat := 95481
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }
def rhsRaw : List Term := Proof.Events372.exact95478RawTerms
def group : MergeGroup := .relation 95480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95480) (rhsResult := 95478)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20093⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95479 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20093⟩⟩]⟩) (none) 95478) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95481

namespace LeftMerge95482
def owner : Owner := ⟨.program ⟨214⟩, ⟨20096⟩⟩
def mergeEvent : Nat := 95482
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩] } }
def rhsRaw : List Term := Proof.Events372.exact95478RawTerms
def group : MergeGroup := .relation 95480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95480) (rhsResult := 95478)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20093⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95479 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20093⟩⟩]⟩) (none) 95478) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95482

namespace LeftMerge95483
def owner : Owner := ⟨.program ⟨214⟩, ⟨20096⟩⟩
def mergeEvent : Nat := 95483
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23326⟩⟩] } }
def rhsRaw : List Term := Proof.Events372.exact95478RawTerms
def group : MergeGroup := .relation 95480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95480) (rhsResult := 95478)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20093⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95479 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20093⟩⟩]⟩) (none) 95478) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23326⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], [⟨.program ⟨214⟩, ⟨23326⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95483

namespace LeftMerge95484
def owner : Owner := ⟨.program ⟨214⟩, ⟨20096⟩⟩
def mergeEvent : Nat := 95484
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events372.exact95478RawTerms
def group : MergeGroup := .relation 95480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95480) (rhsResult := 95478)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20093⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95479 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20093⟩⟩]⟩) (none) 95478) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95484

namespace LeftMerge95489
def owner : Owner := ⟨.program ⟨214⟩, ⟨25593⟩⟩
def mergeEvent : Nat := 95489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23326⟩⟩] } }
def leftRaw : List Term := Proof.Events372.exact95485RawTerms
def rightRaw : List Term := Proof.Events372.exact95323RawTerms
def group : MergeGroup := .operator 95485 95323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95485) (leftOrdinal := 2)
    (rightResult := 95323) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23326⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23326⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], [⟨.program ⟨214⟩, ⟨23326⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95489

namespace LeftMerge95490
def owner : Owner := ⟨.program ⟨214⟩, ⟨25593⟩⟩
def mergeEvent : Nat := 95490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩] } }
def leftRaw : List Term := Proof.Events372.exact95485RawTerms
def rightRaw : List Term := Proof.Events372.exact95323RawTerms
def group : MergeGroup := .operator 95485 95323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95485) (leftOrdinal := 1)
    (rightResult := 95323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95490

namespace LeftMerge95498
def owner : Owner := ⟨.program ⟨214⟩, ⟨29569⟩⟩
def mergeEvent : Nat := 95498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95492RawTerms
def rightRaw : List Term := Proof.Events372.exact95239RawTerms
def group : MergeGroup := .operator 95492 95239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95492) (leftOrdinal := 0)
    (rightResult := 95239) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29567⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95498

namespace LeftMerge95499
def owner : Owner := ⟨.program ⟨214⟩, ⟨29569⟩⟩
def mergeEvent : Nat := 95499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95492RawTerms
def rightRaw : List Term := Proof.Events372.exact95239RawTerms
def group : MergeGroup := .operator 95492 95239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95492) (leftOrdinal := 1)
    (rightResult := 95239) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29567⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95499

namespace LeftMerge95501
def owner : Owner := ⟨.program ⟨214⟩, ⟨29569⟩⟩
def mergeEvent : Nat := 95501
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24657⟩⟩] } }
def rhsRaw : List Term := Proof.Events372.exact95236RawTerms
def group : MergeGroup := .relation 95500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95500) (rhsResult := 95236)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29567⟩⟩) ⟨24657⟩ 95236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24657⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95501

namespace LeftMerge95515
def owner : Owner := ⟨.program ⟨214⟩, ⟨22544⟩⟩
def mergeEvent : Nat := 95515
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22541⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events373.exact95509RawTerms
def group : MergeGroup := .operator 94462 95509
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 95509) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22541⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95515

namespace LeftMerge95612
def owner : Owner := ⟨.program ⟨214⟩, ⟨16821⟩⟩
def mergeEvent : Nat := 95612
def frameStart : Nat := 95558
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95608RawTerms
def rightRaw : List Term := Proof.Events373.exact95606RawTerms
def group : MergeGroup := .operator 95608 95606
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95608) (leftOrdinal := 0)
    (rightResult := 95606) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95612

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
