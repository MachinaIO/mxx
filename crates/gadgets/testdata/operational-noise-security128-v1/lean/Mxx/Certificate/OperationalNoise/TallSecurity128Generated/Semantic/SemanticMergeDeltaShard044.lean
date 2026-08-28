import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge14185
def owner : Owner := ⟨.program ⟨257⟩, ⟨15935⟩⟩
def mergeEvent : Nat := 14185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14181RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 14181 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14181) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15934⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14185

namespace LeftMerge14266
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14266
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 5)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14266

namespace LeftMerge14267
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14267
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 7)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14267

namespace LeftMerge14268
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14268
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 8)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14268

namespace LeftMerge14269
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14269
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 9)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14269

namespace LeftMerge14270
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 11)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14270

namespace LeftMerge14271
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14271
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 12)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14271

namespace LeftMerge14272
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14272
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 13)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14272

namespace LeftMerge14273
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14273
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 15)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14273

namespace LeftMerge14274
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14274
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 16)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14274

namespace LeftMerge14275
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 18)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14275

namespace LeftMerge14276
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 0)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14276

namespace LeftMerge14277
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 1)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14277

namespace LeftMerge14278
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 2)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14278

namespace LeftMerge14279
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 3)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14279

namespace LeftMerge14280
def owner : Owner := ⟨.program ⟨257⟩, ⟨67345⟩⟩
def mergeEvent : Nat := 14280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events055.exact14262RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 14262 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14262) (leftOrdinal := 4)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14280

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
