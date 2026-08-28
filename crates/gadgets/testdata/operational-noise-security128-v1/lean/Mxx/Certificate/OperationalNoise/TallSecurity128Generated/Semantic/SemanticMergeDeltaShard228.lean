import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge42003
def owner : Owner := ⟨.program ⟨257⟩, ⟨69125⟩⟩
def mergeEvent : Nat := 42003
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57292⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact41988RawTerms
def rightRaw : List Term := Proof.Events164.exact41986RawTerms
def group : MergeGroup := .operator 41988 41986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41988) (leftOrdinal := 0)
    (rightResult := 41986) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57292⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57292⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42003

namespace LeftMerge42004
def owner : Owner := ⟨.program ⟨257⟩, ⟨69125⟩⟩
def mergeEvent : Nat := 42004
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54312⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact41988RawTerms
def rightRaw : List Term := Proof.Events164.exact41986RawTerms
def group : MergeGroup := .operator 41988 41986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41988) (leftOrdinal := 0)
    (rightResult := 41986) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54312⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42004

namespace LeftMerge42005
def owner : Owner := ⟨.program ⟨257⟩, ⟨69125⟩⟩
def mergeEvent : Nat := 42005
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51332⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact41988RawTerms
def rightRaw : List Term := Proof.Events164.exact41986RawTerms
def group : MergeGroup := .operator 41988 41986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41988) (leftOrdinal := 0)
    (rightResult := 41986) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51332⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42005

namespace LeftMerge42006
def owner : Owner := ⟨.program ⟨257⟩, ⟨69125⟩⟩
def mergeEvent : Nat := 42006
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32277⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact41988RawTerms
def rightRaw : List Term := Proof.Events164.exact41986RawTerms
def group : MergeGroup := .operator 41988 41986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41988) (leftOrdinal := 0)
    (rightResult := 41986) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32277⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42006

namespace LeftMerge42007
def owner : Owner := ⟨.program ⟨257⟩, ⟨69125⟩⟩
def mergeEvent : Nat := 42007
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22257⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact41988RawTerms
def rightRaw : List Term := Proof.Events164.exact41986RawTerms
def group : MergeGroup := .operator 41988 41986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41988) (leftOrdinal := 0)
    (rightResult := 41986) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22257⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42007

namespace LeftMerge42008
def owner : Owner := ⟨.program ⟨257⟩, ⟨69125⟩⟩
def mergeEvent : Nat := 42008
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨19037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact41988RawTerms
def rightRaw : List Term := Proof.Events164.exact41986RawTerms
def group : MergeGroup := .operator 41988 41986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41988) (leftOrdinal := 0)
    (rightResult := 41986) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨19037⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨19037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42008

namespace LeftMerge42009
def owner : Owner := ⟨.program ⟨257⟩, ⟨69125⟩⟩
def mergeEvent : Nat := 42009
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16179⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact41988RawTerms
def rightRaw : List Term := Proof.Events164.exact41986RawTerms
def group : MergeGroup := .operator 41988 41986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 41988) (leftOrdinal := 0)
    (rightResult := 41986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16179⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42009

namespace LeftMerge42140
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42140
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 17)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42140

namespace LeftMerge42141
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42141
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 16)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42141

namespace LeftMerge42142
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42142
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 15)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42142

namespace LeftMerge42143
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42143
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 14)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42143

namespace LeftMerge42144
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42144
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 13)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42144

namespace LeftMerge42145
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42145
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 12)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42145

namespace LeftMerge42146
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42146
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 11)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42146

namespace LeftMerge42147
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42147
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 10)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42147

namespace LeftMerge42148
def owner : Owner := ⟨.program ⟨257⟩, ⟨71535⟩⟩
def mergeEvent : Nat := 42148
def frameStart : Nat := 41461
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩] } }
def leftRaw : List Term := Proof.Events164.exact42136RawTerms
def rightRaw : List Term := Proof.Events163.exact41977RawTerms
def group : MergeGroup := .operator 42136 41977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42136) (leftOrdinal := 9)
    (rightResult := 41977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71534⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42148

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
