import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge86158
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86158
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 28)
    (rightResult := 84733) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86158

namespace LeftMerge86159
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 14)
    (rightResult := 84733) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86159

namespace LeftMerge86160
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 27)
    (rightResult := 84733) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86160

namespace LeftMerge86161
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 13)
    (rightResult := 84733) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86161

namespace LeftMerge86162
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86162
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 26)
    (rightResult := 84733) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86162

namespace LeftMerge86163
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86163
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 12)
    (rightResult := 84733) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86163

namespace LeftMerge86164
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 25)
    (rightResult := 84733) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86164

namespace LeftMerge86165
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86165
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 11)
    (rightResult := 84733) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86165

namespace LeftMerge86166
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86166
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 23)
    (rightResult := 84733) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86166

namespace LeftMerge86167
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 10)
    (rightResult := 84733) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86167

namespace LeftMerge86168
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 22)
    (rightResult := 84733) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86168

namespace LeftMerge86169
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 9)
    (rightResult := 84733) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86169

namespace LeftMerge86170
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 36)
    (rightResult := 84733) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86170

namespace LeftMerge86171
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 8)
    (rightResult := 84733) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86171

namespace LeftMerge86172
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 35)
    (rightResult := 84733) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86172

namespace LeftMerge86173
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86173
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 7)
    (rightResult := 84733) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86173

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
