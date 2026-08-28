import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge148777
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 9)
    (rightResult := 134368) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45588⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148777

namespace LeftMerge148778
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 10)
    (rightResult := 134368) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148778

namespace LeftMerge148779
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 12)
    (rightResult := 134368) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148779

namespace LeftMerge148780
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148780
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 13)
    (rightResult := 134368) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148780

namespace LeftMerge148781
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148781
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 14)
    (rightResult := 134368) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148781

namespace LeftMerge148782
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148782
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 16)
    (rightResult := 134368) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148782

namespace LeftMerge148783
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148783
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 17)
    (rightResult := 134368) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148783

namespace LeftMerge148784
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148784
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 19)
    (rightResult := 134368) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148784

namespace LeftMerge148785
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148785
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 1)
    (rightResult := 134368) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148785

namespace LeftMerge148786
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148786
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 2)
    (rightResult := 134368) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148786

namespace LeftMerge148787
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148787
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 3)
    (rightResult := 134368) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148787

namespace LeftMerge148788
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148788
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 4)
    (rightResult := 134368) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148788

namespace LeftMerge148789
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 5)
    (rightResult := 134368) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148789

namespace LeftMerge148790
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 7)
    (rightResult := 134368) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148790

namespace LeftMerge148791
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148791
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 11)
    (rightResult := 134368) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148791

namespace LeftMerge148792
def owner : Owner := ⟨.program ⟨257⟩, ⟨71026⟩⟩
def mergeEvent : Nat := 148792
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148771RawTerms
def rightRaw : List Term := Proof.Events524.exact134368RawTerms
def group : MergeGroup := .operator 148771 134368
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148771) (leftOrdinal := 15)
    (rightResult := 134368) (rightOrdinal := 33) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7251⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], [⟨.program ⟨257⟩, ⟨7251⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148792

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
