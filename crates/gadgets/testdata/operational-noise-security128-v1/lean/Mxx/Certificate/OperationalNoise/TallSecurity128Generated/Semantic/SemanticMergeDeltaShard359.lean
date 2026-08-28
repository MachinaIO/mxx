import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge61225
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61225

namespace LeftMerge61226
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61226
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61226

namespace LeftMerge61227
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61227
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61227

namespace LeftMerge61228
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61228
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61228

namespace LeftMerge61229
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61229
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61229

namespace LeftMerge61230
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61230
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61230

namespace LeftMerge61231
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61231

namespace LeftMerge61232
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61232

namespace LeftMerge61233
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61233
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61233

namespace LeftMerge61234
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61234
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61234

namespace LeftMerge61235
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61235
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61235

namespace LeftMerge61236
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61236
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61236

namespace LeftMerge61237
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61237
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61237

namespace LeftMerge61238
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61238
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61238

namespace LeftMerge61239
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61239
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61239

namespace LeftMerge61240
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def mergeEvent : Nat := 61240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61163RawTerms
def rightRaw : List Term := Proof.Events011.exact3048RawTerms
def group : MergeGroup := .operator 61163 3048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61163) (leftOrdinal := 1)
    (rightResult := 3048) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7241⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61240

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
