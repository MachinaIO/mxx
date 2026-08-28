import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge86174
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86174
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 34)
    (rightResult := 84733) (rightOrdinal := 33) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86174

namespace LeftMerge86175
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86175
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 6)
    (rightResult := 84733) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86175

namespace LeftMerge86176
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 33)
    (rightResult := 84733) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86176

namespace LeftMerge86177
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 5)
    (rightResult := 84733) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86177

namespace LeftMerge86178
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 32)
    (rightResult := 84733) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86178

namespace LeftMerge86179
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86179
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 4)
    (rightResult := 84733) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86179

namespace LeftMerge86180
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86180
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 31)
    (rightResult := 84733) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86180

namespace LeftMerge86181
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86181
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 3)
    (rightResult := 84733) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86181

namespace LeftMerge86182
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86182
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 24)
    (rightResult := 84733) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86182

namespace LeftMerge86183
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86183
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 2)
    (rightResult := 84733) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86183

namespace LeftMerge86184
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 21)
    (rightResult := 84733) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86184

namespace LeftMerge86185
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 1)
    (rightResult := 84733) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86185

namespace LeftMerge86186
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 20)
    (rightResult := 84733) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86186

namespace LeftMerge86187
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 0)
    (rightResult := 84733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86187

namespace LeftMerge86188
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def mergeEvent : Nat := 86188
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86149RawTerms
def rightRaw : List Term := Proof.Events330.exact84733RawTerms
def group : MergeGroup := .operator 86149 84733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86149) (leftOrdinal := 19)
    (rightResult := 84733) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86188

namespace LeftMerge86196
def owner : Owner := ⟨.program ⟨257⟩, ⟨71441⟩⟩
def mergeEvent : Nat := 86196
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86190RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 86190 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86190) (leftOrdinal := 0)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86196

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
