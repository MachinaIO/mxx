import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge159141
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159141
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 16)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159141

namespace LeftMerge159142
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159142
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 15)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159142

namespace LeftMerge159143
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159143
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 14)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159143

namespace LeftMerge159144
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159144
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 13)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159144

namespace LeftMerge159145
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159145
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 12)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159145

namespace LeftMerge159146
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159146
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 11)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159146

namespace LeftMerge159147
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159147
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 10)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159147

namespace LeftMerge159148
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159148
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 9)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159148

namespace LeftMerge159149
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159149
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 8)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159149

namespace LeftMerge159150
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159150
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 7)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159150

namespace LeftMerge159151
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159151
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 6)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159151

namespace LeftMerge159152
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159152
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 5)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159152

namespace LeftMerge159153
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159153
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 4)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159153

namespace LeftMerge159154
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159154
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 3)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159154

namespace LeftMerge159155
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159155
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 2)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159155

namespace LeftMerge159156
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159156
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 1)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159156

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
