import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge285501
def owner : Owner := ⟨.program ⟨257⟩, ⟨7913⟩⟩
def mergeEvent : Nat := 285501
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280523RawTerms
def rightRaw : List Term := Proof.Events086.exact22131RawTerms
def group : MergeGroup := .operator 280523 22131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280523) (leftOrdinal := 0)
    (rightResult := 22131) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285501

namespace LeftMerge285518
def owner : Owner := ⟨.program ⟨257⟩, ⟨59330⟩⟩
def mergeEvent : Nat := 285518
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285512RawTerms
def rightRaw : List Term := Proof.Events086.exact22120RawTerms
def group : MergeGroup := .operator 285512 22120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285512) (leftOrdinal := 1)
    (rightResult := 22120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285518

namespace LeftMerge285520
def owner : Owner := ⟨.program ⟨257⟩, ⟨59330⟩⟩
def mergeEvent : Nat := 285520
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def rhsRaw : List Term := Proof.Events086.exact22090RawTerms
def group : MergeGroup := .relation 285519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285519) (rhsResult := 22090)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9535⟩⟩) ⟨7274⟩ 22090) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285520

namespace LeftMerge285521
def owner : Owner := ⟨.program ⟨257⟩, ⟨59330⟩⟩
def mergeEvent : Nat := 285521
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285512RawTerms
def rightRaw : List Term := Proof.Events086.exact22120RawTerms
def group : MergeGroup := .operator 285512 22120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285512) (leftOrdinal := 0)
    (rightResult := 22120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285521

namespace LeftMerge285526
def owner : Owner := ⟨.program ⟨257⟩, ⟨59331⟩⟩
def mergeEvent : Nat := 285526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285522RawTerms
def rightRaw : List Term := Proof.Events1115.exact285492RawTerms
def group : MergeGroup := .operator 285522 285492
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285522) (leftOrdinal := 1)
    (rightResult := 285492) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285526

namespace LeftMerge285534
def owner : Owner := ⟨.program ⟨257⟩, ⟨61394⟩⟩
def mergeEvent : Nat := 285534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285528RawTerms
def rightRaw : List Term := Proof.Events1115.exact285464RawTerms
def group : MergeGroup := .operator 285528 285464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285528) (leftOrdinal := 1)
    (rightResult := 285464) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61393⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285534

namespace LeftMerge285536
def owner : Owner := ⟨.program ⟨257⟩, ⟨61394⟩⟩
def mergeEvent : Nat := 285536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60913⟩⟩] } }
def rhsRaw : List Term := Proof.Events1115.exact285461RawTerms
def group : MergeGroup := .relation 285535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285535) (rhsResult := 285461)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61393⟩⟩) ⟨60913⟩ 285461) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60913⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨60913⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285536

namespace LeftMerge285537
def owner : Owner := ⟨.program ⟨257⟩, ⟨61394⟩⟩
def mergeEvent : Nat := 285537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285528RawTerms
def rightRaw : List Term := Proof.Events1115.exact285464RawTerms
def group : MergeGroup := .operator 285528 285464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285528) (leftOrdinal := 0)
    (rightResult := 285464) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61393⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285537

namespace LeftMerge285551
def owner : Owner := ⟨.program ⟨257⟩, ⟨60332⟩⟩
def mergeEvent : Nat := 285551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60329⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1115.exact285545RawTerms
def group : MergeGroup := .operator 280745 285545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 285545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285551

namespace LeftMerge285630
def owner : Owner := ⟨.program ⟨257⟩, ⟨59324⟩⟩
def mergeEvent : Nat := 285630
def frameStart : Nat := 285600
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1115.exact285626RawTerms
def rightRaw : List Term := Proof.Events1115.exact285623RawTerms
def group : MergeGroup := .operator 285626 285623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285626) (leftOrdinal := 0)
    (rightResult := 285623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25178⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285630

namespace LeftMerge285660
def owner : Owner := ⟨.program ⟨257⟩, ⟨61204⟩⟩
def mergeEvent : Nat := 285660
def frameStart : Nat := 285600
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285656RawTerms
def rightRaw : List Term := Proof.Events1115.exact285654RawTerms
def group : MergeGroup := .operator 285656 285654
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285656) (leftOrdinal := 0)
    (rightResult := 285654) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285660

namespace LeftMerge285681
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 285681
def frameStart : Nat := 285600
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285677RawTerms
def rightRaw : List Term := Proof.Events1115.exact285674RawTerms
def group : MergeGroup := .operator 285677 285674
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285677) (leftOrdinal := 0)
    (rightResult := 285674) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285681

namespace LeftMerge285690
def owner : Owner := ⟨.program ⟨257⟩, ⟨61396⟩⟩
def mergeEvent : Nat := 285690
def frameStart : Nat := 285600
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285686RawTerms
def rightRaw : List Term := Proof.Events1115.exact285645RawTerms
def group : MergeGroup := .operator 285686 285645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285686) (leftOrdinal := 0)
    (rightResult := 285645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61393⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285690

namespace LeftMerge285691
def owner : Owner := ⟨.program ⟨257⟩, ⟨61396⟩⟩
def mergeEvent : Nat := 285691
def frameStart : Nat := 285600
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285686RawTerms
def rightRaw : List Term := Proof.Events1115.exact285645RawTerms
def group : MergeGroup := .operator 285686 285645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285686) (leftOrdinal := 1)
    (rightResult := 285645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61393⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285691

namespace LeftMerge285693
def owner : Owner := ⟨.program ⟨257⟩, ⟨61396⟩⟩
def mergeEvent : Nat := 285693
def frameStart : Nat := 285600
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60913⟩⟩] } }
def rhsRaw : List Term := Proof.Events1115.exact285642RawTerms
def group : MergeGroup := .relation 285692
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285692) (rhsResult := 285642)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61393⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61393⟩⟩) ⟨60913⟩ 285642) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60913⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨60913⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285693

namespace LeftMerge285701
def owner : Owner := ⟨.program ⟨257⟩, ⟨59782⟩⟩
def mergeEvent : Nat := 285701
def frameStart : Nat := 285600
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285656RawTerms
def rightRaw : List Term := Proof.Events1116.exact285697RawTerms
def group : MergeGroup := .operator 285656 285697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285656) (leftOrdinal := 0)
    (rightResult := 285697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59780⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285701

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
