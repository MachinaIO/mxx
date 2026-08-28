import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge134145
def owner : Owner := ⟨.program ⟨257⟩, ⟨71121⟩⟩
def mergeEvent : Nat := 134145
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9505⟩⟩] } }
def leftRaw : List Term := Proof.Events523.exact134082RawTerms
def rightRaw : List Term := Proof.Events063.exact16264RawTerms
def group : MergeGroup := .operator 134082 16264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134082) (leftOrdinal := 0)
    (rightResult := 16264) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9505⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9505⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134145

namespace LeftMerge134150
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134150
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 6)
    (rightResult := 119743) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge134150

namespace LeftMerge134151
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 8)
    (rightResult := 119743) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134151

namespace LeftMerge134152
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 9)
    (rightResult := 119743) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134152

namespace LeftMerge134153
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 10)
    (rightResult := 119743) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134153

namespace LeftMerge134154
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 12)
    (rightResult := 119743) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134154

namespace LeftMerge134155
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134155
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 13)
    (rightResult := 119743) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134155

namespace LeftMerge134156
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134156
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 14)
    (rightResult := 119743) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134156

namespace LeftMerge134157
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134157
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 16)
    (rightResult := 119743) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134157

namespace LeftMerge134158
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134158
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 17)
    (rightResult := 119743) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134158

namespace LeftMerge134159
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 19)
    (rightResult := 119743) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134159

namespace LeftMerge134160
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 1)
    (rightResult := 119743) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134160

namespace LeftMerge134161
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 2)
    (rightResult := 119743) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134161

namespace LeftMerge134162
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134162
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 3)
    (rightResult := 119743) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134162

namespace LeftMerge134163
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134163
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 4)
    (rightResult := 119743) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134163

namespace LeftMerge134164
def owner : Owner := ⟨.program ⟨257⟩, ⟨71122⟩⟩
def mergeEvent : Nat := 134164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134146RawTerms
def rightRaw : List Term := Proof.Events467.exact119743RawTerms
def group : MergeGroup := .operator 134146 119743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134146) (leftOrdinal := 5)
    (rightResult := 119743) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], [⟨.program ⟨257⟩, ⟨7249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134164

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
