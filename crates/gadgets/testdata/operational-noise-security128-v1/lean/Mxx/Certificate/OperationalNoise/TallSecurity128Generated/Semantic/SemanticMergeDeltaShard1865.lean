import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge301430
def owner : Owner := ⟨.program ⟨257⟩, ⟨33351⟩⟩
def mergeEvent : Nat := 301430
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32889⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301426RawTerms
def rightRaw : List Term := Proof.Events1176.exact301264RawTerms
def group : MergeGroup := .operator 301426 301264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301426) (leftOrdinal := 2)
    (rightResult := 301264) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32889⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32889⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24170⟩⟩, ⟨.program ⟨257⟩, ⟨31215⟩⟩], [⟨.program ⟨257⟩, ⟨32889⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301430

namespace LeftMerge301431
def owner : Owner := ⟨.program ⟨257⟩, ⟨33351⟩⟩
def mergeEvent : Nat := 301431
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301426RawTerms
def rightRaw : List Term := Proof.Events1176.exact301264RawTerms
def group : MergeGroup := .operator 301426 301264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301426) (leftOrdinal := 1)
    (rightResult := 301264) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301431

namespace LeftMerge301439
def owner : Owner := ⟨.program ⟨257⟩, ⟨33584⟩⟩
def mergeEvent : Nat := 301439
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301433RawTerms
def rightRaw : List Term := Proof.Events1176.exact301180RawTerms
def group : MergeGroup := .operator 301433 301180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301433) (leftOrdinal := 0)
    (rightResult := 301180) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301439

namespace LeftMerge301440
def owner : Owner := ⟨.program ⟨257⟩, ⟨33584⟩⟩
def mergeEvent : Nat := 301440
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301433RawTerms
def rightRaw : List Term := Proof.Events1176.exact301180RawTerms
def group : MergeGroup := .operator 301433 301180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301433) (leftOrdinal := 1)
    (rightResult := 301180) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301440

namespace LeftMerge301442
def owner : Owner := ⟨.program ⟨257⟩, ⟨33584⟩⟩
def mergeEvent : Nat := 301442
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33011⟩⟩] } }
def rhsRaw : List Term := Proof.Events1176.exact301177RawTerms
def group : MergeGroup := .relation 301441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301441) (rhsResult := 301177)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33582⟩⟩) ⟨33011⟩ 301177) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33011⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨33011⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301442

namespace LeftMerge301456
def owner : Owner := ⟨.program ⟨257⟩, ⟨32499⟩⟩
def mergeEvent : Nat := 301456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1177.exact301450RawTerms
def group : MergeGroup := .operator 295195 301450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 301450) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32496⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301456

namespace LeftMerge301553
def owner : Owner := ⟨.program ⟨257⟩, ⟨33268⟩⟩
def mergeEvent : Nat := 301553
def frameStart : Nat := 301499
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301549RawTerms
def rightRaw : List Term := Proof.Events1177.exact301547RawTerms
def group : MergeGroup := .operator 301549 301547
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301549) (leftOrdinal := 0)
    (rightResult := 301547) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301553

namespace LeftMerge301565
def owner : Owner := ⟨.program ⟨257⟩, ⟨33583⟩⟩
def mergeEvent : Nat := 301565
def frameStart : Nat := 301499
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301561RawTerms
def rightRaw : List Term := Proof.Events1177.exact301538RawTerms
def group : MergeGroup := .operator 301561 301538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301561) (leftOrdinal := 0)
    (rightResult := 301538) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33582⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301565

namespace LeftMerge301566
def owner : Owner := ⟨.program ⟨257⟩, ⟨33583⟩⟩
def mergeEvent : Nat := 301566
def frameStart : Nat := 301499
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301561RawTerms
def rightRaw : List Term := Proof.Events1177.exact301538RawTerms
def group : MergeGroup := .operator 301561 301538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301561) (leftOrdinal := 1)
    (rightResult := 301538) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301566

namespace LeftMerge301568
def owner : Owner := ⟨.program ⟨257⟩, ⟨33583⟩⟩
def mergeEvent : Nat := 301568
def frameStart : Nat := 301499
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33011⟩⟩] } }
def rhsRaw : List Term := Proof.Events1177.exact301535RawTerms
def group : MergeGroup := .relation 301567
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301567) (rhsResult := 301535)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33582⟩⟩) ⟨33011⟩ 301535) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33011⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨33011⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301568

namespace LeftMerge301576
def owner : Owner := ⟨.program ⟨257⟩, ⟨31918⟩⟩
def mergeEvent : Nat := 301576
def frameStart : Nat := 301499
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1177.exact301549RawTerms
def rightRaw : List Term := Proof.Events1178.exact301572RawTerms
def group : MergeGroup := .operator 301549 301572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301549) (leftOrdinal := 0)
    (rightResult := 301572) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31916⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301576

namespace LeftMerge301593
def owner : Owner := ⟨.program ⟨257⟩, ⟨32499⟩⟩
def mergeEvent : Nat := 301593
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }
def rhsRaw : List Term := Proof.Events1178.exact301590RawTerms
def group : MergeGroup := .relation 301592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301592) (rhsResult := 301590)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301591 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩) (none) 301590) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301593

namespace LeftMerge301594
def owner : Owner := ⟨.program ⟨257⟩, ⟨32499⟩⟩
def mergeEvent : Nat := 301594
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩] } }
def rhsRaw : List Term := Proof.Events1178.exact301590RawTerms
def group : MergeGroup := .relation 301592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301592) (rhsResult := 301590)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301591 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩) (none) 301590) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301594

namespace LeftMerge301595
def owner : Owner := ⟨.program ⟨257⟩, ⟨32499⟩⟩
def mergeEvent : Nat := 301595
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33011⟩⟩] } }
def rhsRaw : List Term := Proof.Events1178.exact301590RawTerms
def group : MergeGroup := .relation 301592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301592) (rhsResult := 301590)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301591 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩) (none) 301590) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33011⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31748⟩⟩], [⟨.program ⟨257⟩, ⟨33011⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301595

namespace LeftMerge301596
def owner : Owner := ⟨.program ⟨257⟩, ⟨32499⟩⟩
def mergeEvent : Nat := 301596
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1178.exact301590RawTerms
def group : MergeGroup := .relation 301592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301592) (rhsResult := 301590)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301591 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩) (none) 301590) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301596

namespace LeftMerge301601
def owner : Owner := ⟨.program ⟨257⟩, ⟨33585⟩⟩
def mergeEvent : Nat := 301601
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩] } }
def leftRaw : List Term := Proof.Events1178.exact301597RawTerms
def rightRaw : List Term := Proof.Events1177.exact301443RawTerms
def group : MergeGroup := .operator 301597 301443
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301597) (leftOrdinal := 0)
    (rightResult := 301443) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33582⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301601

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
