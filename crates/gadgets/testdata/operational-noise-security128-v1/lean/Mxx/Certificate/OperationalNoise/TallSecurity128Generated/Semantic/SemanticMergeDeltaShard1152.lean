import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge188518
def owner : Owner := ⟨.program ⟨257⟩, ⟨68403⟩⟩
def mergeEvent : Nat := 188518
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events736.exact188483RawTerms
def group : MergeGroup := .relation 188485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188485) (rhsResult := 188483)
    (sourceTermOrdinal := 31) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188484 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩) (none) 188483) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188518

namespace LeftMerge188519
def owner : Owner := ⟨.program ⟨257⟩, ⟨68403⟩⟩
def mergeEvent : Nat := 188519
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events736.exact188483RawTerms
def group : MergeGroup := .relation 188485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188485) (rhsResult := 188483)
    (sourceTermOrdinal := 24) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188484 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩) (none) 188483) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188519

namespace LeftMerge188520
def owner : Owner := ⟨.program ⟨257⟩, ⟨68403⟩⟩
def mergeEvent : Nat := 188520
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events736.exact188483RawTerms
def group : MergeGroup := .relation 188485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188485) (rhsResult := 188483)
    (sourceTermOrdinal := 21) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188484 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩) (none) 188483) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188520

namespace LeftMerge188521
def owner : Owner := ⟨.program ⟨257⟩, ⟨68403⟩⟩
def mergeEvent : Nat := 188521
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events736.exact188483RawTerms
def group : MergeGroup := .relation 188485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188485) (rhsResult := 188483)
    (sourceTermOrdinal := 20) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188484 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩) (none) 188483) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18923⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188521

namespace LeftMerge188522
def owner : Owner := ⟨.program ⟨257⟩, ⟨68403⟩⟩
def mergeEvent : Nat := 188522
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events736.exact188483RawTerms
def group : MergeGroup := .relation 188485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188485) (rhsResult := 188483)
    (sourceTermOrdinal := 19) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188484 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩) (none) 188483) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188522

namespace LeftMerge188523
def owner : Owner := ⟨.program ⟨257⟩, ⟨68403⟩⟩
def mergeEvent : Nat := 188523
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨67514⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events736.exact188483RawTerms
def group : MergeGroup := .relation 188485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188485) (rhsResult := 188483)
    (sourceTermOrdinal := 37) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 188484 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩) (none) 188483) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67514⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨67514⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188523

namespace LeftMerge188528
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 17)
    (rightResult := 187108) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188528

namespace LeftMerge188529
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 30)
    (rightResult := 187108) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188529

namespace LeftMerge188530
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188530
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 16)
    (rightResult := 187108) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188530

namespace LeftMerge188531
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188531
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 29)
    (rightResult := 187108) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188531

namespace LeftMerge188532
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188532
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 15)
    (rightResult := 187108) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188532

namespace LeftMerge188533
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 28)
    (rightResult := 187108) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188533

namespace LeftMerge188534
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 14)
    (rightResult := 187108) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188534

namespace LeftMerge188535
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 27)
    (rightResult := 187108) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188535

namespace LeftMerge188536
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 13)
    (rightResult := 187108) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188536

namespace LeftMerge188537
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 26)
    (rightResult := 187108) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188537

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
