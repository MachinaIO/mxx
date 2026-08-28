import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge297491
def owner : Owner := ⟨.program ⟨257⟩, ⟨36152⟩⟩
def mergeEvent : Nat := 297491
def frameStart : Nat := 297408
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35689⟩⟩] } }
def rhsRaw : List Term := Proof.Events1161.exact297438RawTerms
def group : MergeGroup := .relation 297490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297490) (rhsResult := 297438)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36149⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36149⟩⟩) ⟨35689⟩ 297438) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35689⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], [⟨.program ⟨257⟩, ⟨35689⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297491

namespace LeftMerge297499
def owner : Owner := ⟨.program ⟨257⟩, ⟨34670⟩⟩
def mergeEvent : Nat := 297499
def frameStart : Nat := 297408
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1161.exact297452RawTerms
def rightRaw : List Term := Proof.Events1162.exact297495RawTerms
def group : MergeGroup := .operator 297452 297495
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297452) (leftOrdinal := 0)
    (rightResult := 297495) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34668⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297499

namespace LeftMerge297516
def owner : Owner := ⟨.program ⟨257⟩, ⟨35092⟩⟩
def mergeEvent : Nat := 297516
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events1162.exact297513RawTerms
def group : MergeGroup := .relation 297515
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297515) (rhsResult := 297513)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35089⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297514 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35089⟩⟩]⟩) (none) 297513) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297516

namespace LeftMerge297517
def owner : Owner := ⟨.program ⟨257⟩, ⟨35092⟩⟩
def mergeEvent : Nat := 297517
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36149⟩⟩] } }
def rhsRaw : List Term := Proof.Events1162.exact297513RawTerms
def group : MergeGroup := .relation 297515
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297515) (rhsResult := 297513)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35089⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297514 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35089⟩⟩]⟩) (none) 297513) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36149⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36149⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297517

namespace LeftMerge297518
def owner : Owner := ⟨.program ⟨257⟩, ⟨35092⟩⟩
def mergeEvent : Nat := 297518
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35689⟩⟩] } }
def rhsRaw : List Term := Proof.Events1162.exact297513RawTerms
def group : MergeGroup := .relation 297515
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297515) (rhsResult := 297513)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35089⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297514 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35089⟩⟩]⟩) (none) 297513) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35689⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], [⟨.program ⟨257⟩, ⟨35689⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297518

namespace LeftMerge297519
def owner : Owner := ⟨.program ⟨257⟩, ⟨35092⟩⟩
def mergeEvent : Nat := 297519
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1162.exact297513RawTerms
def group : MergeGroup := .relation 297515
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297515) (rhsResult := 297513)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35089⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297514 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35089⟩⟩]⟩) (none) 297513) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34668⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297519

namespace LeftMerge297524
def owner : Owner := ⟨.program ⟨257⟩, ⟨36151⟩⟩
def mergeEvent : Nat := 297524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35689⟩⟩] } }
def leftRaw : List Term := Proof.Events1162.exact297520RawTerms
def rightRaw : List Term := Proof.Events1161.exact297358RawTerms
def group : MergeGroup := .operator 297520 297358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297520) (leftOrdinal := 2)
    (rightResult := 297358) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35689⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35689⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13431⟩⟩, ⟨.program ⟨257⟩, ⟨34194⟩⟩], [⟨.program ⟨257⟩, ⟨35689⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297524

namespace LeftMerge297525
def owner : Owner := ⟨.program ⟨257⟩, ⟨36151⟩⟩
def mergeEvent : Nat := 297525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36149⟩⟩] } }
def leftRaw : List Term := Proof.Events1162.exact297520RawTerms
def rightRaw : List Term := Proof.Events1161.exact297358RawTerms
def group : MergeGroup := .operator 297520 297358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297520) (leftOrdinal := 1)
    (rightResult := 297358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36149⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36149⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36149⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297525

namespace LeftMerge297533
def owner : Owner := ⟨.program ⟨257⟩, ⟨36381⟩⟩
def mergeEvent : Nat := 297533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩] } }
def leftRaw : List Term := Proof.Events1162.exact297527RawTerms
def rightRaw : List Term := Proof.Events1161.exact297274RawTerms
def group : MergeGroup := .operator 297527 297274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297527) (leftOrdinal := 0)
    (rightResult := 297274) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297533

namespace LeftMerge297534
def owner : Owner := ⟨.program ⟨257⟩, ⟨36381⟩⟩
def mergeEvent : Nat := 297534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩] } }
def leftRaw : List Term := Proof.Events1162.exact297527RawTerms
def rightRaw : List Term := Proof.Events1161.exact297274RawTerms
def group : MergeGroup := .operator 297527 297274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297527) (leftOrdinal := 1)
    (rightResult := 297274) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34668⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297534

namespace LeftMerge297536
def owner : Owner := ⟨.program ⟨257⟩, ⟨36381⟩⟩
def mergeEvent : Nat := 297536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35811⟩⟩] } }
def rhsRaw : List Term := Proof.Events1161.exact297271RawTerms
def group : MergeGroup := .relation 297535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297535) (rhsResult := 297271)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34668⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36379⟩⟩) ⟨35811⟩ 297271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35811⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34668⟩⟩], [⟨.program ⟨257⟩, ⟨35811⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297536

namespace LeftMerge297550
def owner : Owner := ⟨.program ⟨257⟩, ⟨35299⟩⟩
def mergeEvent : Nat := 297550
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35296⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1162.exact297544RawTerms
def group : MergeGroup := .operator 295195 297544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 297544) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35296⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297550

namespace LeftMerge297647
def owner : Owner := ⟨.program ⟨257⟩, ⟨36068⟩⟩
def mergeEvent : Nat := 297647
def frameStart : Nat := 297593
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1162.exact297643RawTerms
def rightRaw : List Term := Proof.Events1162.exact297641RawTerms
def group : MergeGroup := .operator 297643 297641
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297643) (leftOrdinal := 0)
    (rightResult := 297641) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34668⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297647

namespace LeftMerge297659
def owner : Owner := ⟨.program ⟨257⟩, ⟨36380⟩⟩
def mergeEvent : Nat := 297659
def frameStart : Nat := 297593
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩] } }
def leftRaw : List Term := Proof.Events1162.exact297655RawTerms
def rightRaw : List Term := Proof.Events1162.exact297632RawTerms
def group : MergeGroup := .operator 297655 297632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297655) (leftOrdinal := 0)
    (rightResult := 297632) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36379⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297659

namespace LeftMerge297660
def owner : Owner := ⟨.program ⟨257⟩, ⟨36380⟩⟩
def mergeEvent : Nat := 297660
def frameStart : Nat := 297593
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩] } }
def leftRaw : List Term := Proof.Events1162.exact297655RawTerms
def rightRaw : List Term := Proof.Events1162.exact297632RawTerms
def group : MergeGroup := .operator 297655 297632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297655) (leftOrdinal := 1)
    (rightResult := 297632) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34668⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297660

namespace LeftMerge297662
def owner : Owner := ⟨.program ⟨257⟩, ⟨36380⟩⟩
def mergeEvent : Nat := 297662
def frameStart : Nat := 297593
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34668⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35811⟩⟩] } }
def rhsRaw : List Term := Proof.Events1162.exact297629RawTerms
def group : MergeGroup := .relation 297661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297661) (rhsResult := 297629)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34668⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36379⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36379⟩⟩) ⟨35811⟩ 297629) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35811⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34668⟩⟩], [⟨.program ⟨257⟩, ⟨35811⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297662

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
