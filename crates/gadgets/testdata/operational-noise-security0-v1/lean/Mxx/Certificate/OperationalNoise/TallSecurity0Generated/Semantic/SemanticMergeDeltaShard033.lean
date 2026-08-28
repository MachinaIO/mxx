import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge6388
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6388
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 0)
    (rightResult := 804) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6388

namespace LeftMerge6389
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6389
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 0)
    (rightResult := 804) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6389

namespace LeftMerge6390
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6390
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6390

namespace LeftMerge6391
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6391
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6391

namespace LeftMerge6392
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6392

namespace LeftMerge6393
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6393
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6393

namespace LeftMerge6394
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6394
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6394

namespace LeftMerge6395
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6395
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6395

namespace LeftMerge6396
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6396
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6396

namespace LeftMerge6397
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6397
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6397

namespace LeftMerge6398
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6398
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6398

namespace LeftMerge6399
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6399
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6399

namespace LeftMerge6400
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6400

namespace LeftMerge6401
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6401

namespace LeftMerge6402
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6402

namespace LeftMerge6403
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def mergeEvent : Nat := 6403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6329RawTerms
def rightRaw : List Term := Proof.Events003.exact804RawTerms
def group : MergeGroup := .operator 6329 804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6329) (leftOrdinal := 1)
    (rightResult := 804) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6403

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
