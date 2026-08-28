import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge21167
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 6)
    (rightResult := 6409) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21167

namespace LeftMerge21168
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 8)
    (rightResult := 6409) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21168

namespace LeftMerge21169
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 9)
    (rightResult := 6409) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21169

namespace LeftMerge21170
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 10)
    (rightResult := 6409) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21170

namespace LeftMerge21171
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 12)
    (rightResult := 6409) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21171

namespace LeftMerge21172
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 13)
    (rightResult := 6409) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21172

namespace LeftMerge21173
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21173
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 14)
    (rightResult := 6409) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21173

namespace LeftMerge21174
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21174
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 16)
    (rightResult := 6409) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21174

namespace LeftMerge21175
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21175
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 17)
    (rightResult := 6409) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21175

namespace LeftMerge21176
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 19)
    (rightResult := 6409) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21176

namespace LeftMerge21177
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 1)
    (rightResult := 6409) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21177

namespace LeftMerge21178
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 2)
    (rightResult := 6409) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21178

namespace LeftMerge21179
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21179
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 3)
    (rightResult := 6409) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21179

namespace LeftMerge21180
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21180
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 4)
    (rightResult := 6409) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21180

namespace LeftMerge21181
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21181
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 5)
    (rightResult := 6409) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21181

namespace LeftMerge21182
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def mergeEvent : Nat := 21182
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events082.exact21163RawTerms
def rightRaw : List Term := Proof.Events025.exact6409RawTerms
def group : MergeGroup := .operator 21163 6409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21163) (leftOrdinal := 7)
    (rightResult := 6409) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21182

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
