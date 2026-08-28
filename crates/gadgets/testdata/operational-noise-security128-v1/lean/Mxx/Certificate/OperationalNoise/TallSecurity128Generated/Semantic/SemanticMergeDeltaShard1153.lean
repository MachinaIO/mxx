import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge188538
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 12)
    (rightResult := 187108) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188538

namespace LeftMerge188539
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 25)
    (rightResult := 187108) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188539

namespace LeftMerge188540
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 11)
    (rightResult := 187108) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188540

namespace LeftMerge188541
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 23)
    (rightResult := 187108) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188541

namespace LeftMerge188542
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 10)
    (rightResult := 187108) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188542

namespace LeftMerge188543
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 22)
    (rightResult := 187108) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188543

namespace LeftMerge188544
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 9)
    (rightResult := 187108) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188544

namespace LeftMerge188545
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 36)
    (rightResult := 187108) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188545

namespace LeftMerge188546
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 8)
    (rightResult := 187108) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188546

namespace LeftMerge188547
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 35)
    (rightResult := 187108) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188547

namespace LeftMerge188548
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188548
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 7)
    (rightResult := 187108) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188548

namespace LeftMerge188549
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188549
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 34)
    (rightResult := 187108) (rightOrdinal := 33) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188549

namespace LeftMerge188550
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188550
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 6)
    (rightResult := 187108) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188550

namespace LeftMerge188551
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 33)
    (rightResult := 187108) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188551

namespace LeftMerge188552
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 5)
    (rightResult := 187108) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188552

namespace LeftMerge188553
def owner : Owner := ⟨.program ⟨257⟩, ⟨71332⟩⟩
def mergeEvent : Nat := 188553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def leftRaw : List Term := Proof.Events736.exact188524RawTerms
def rightRaw : List Term := Proof.Events730.exact187108RawTerms
def group : MergeGroup := .operator 188524 187108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188524) (leftOrdinal := 32)
    (rightResult := 187108) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188553

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
