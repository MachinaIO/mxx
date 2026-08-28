import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge16321
def owner : Owner := ⟨.program ⟨257⟩, ⟨9673⟩⟩
def mergeEvent : Nat := 16321
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9507⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16317RawTerms
def rightRaw : List Term := Proof.Events063.exact16294RawTerms
def group : MergeGroup := .operator 16317 16294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16317) (leftOrdinal := 0)
    (rightResult := 16294) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9507⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7117⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9507⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16321

namespace LeftMerge16326
def owner : Owner := ⟨.program ⟨257⟩, ⟨7033⟩⟩
def mergeEvent : Nat := 16326
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 2 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16326

namespace LeftMerge16351
def owner : Owner := ⟨.program ⟨257⟩, ⟨9594⟩⟩
def mergeEvent : Nat := 16351
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16347RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16347 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16347) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7252⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16351

namespace LeftMerge16356
def owner : Owner := ⟨.program ⟨257⟩, ⟨9655⟩⟩
def mergeEvent : Nat := 16356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9509⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16352RawTerms
def rightRaw : List Term := Proof.Events063.exact16344RawTerms
def group : MergeGroup := .operator 16352 16344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16352) (leftOrdinal := 0)
    (rightResult := 16344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9509⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9509⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16356

namespace LeftMerge16361
def owner : Owner := ⟨.program ⟨257⟩, ⟨9674⟩⟩
def mergeEvent : Nat := 16361
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9509⟩⟩, ⟨.program ⟨257⟩, ⟨7135⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16357RawTerms
def rightRaw : List Term := Proof.Events063.exact16334RawTerms
def group : MergeGroup := .operator 16357 16334
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16357) (leftOrdinal := 0)
    (rightResult := 16334) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9509⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7135⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9509⟩⟩, ⟨.program ⟨257⟩, ⟨7135⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16361

namespace LeftMerge16366
def owner : Owner := ⟨.program ⟨257⟩, ⟨7029⟩⟩
def mergeEvent : Nat := 16366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events029.exact7561RawTerms
def group : MergeGroup := .operator 2 7561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 7561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6765⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16366

namespace LeftMerge16391
def owner : Owner := ⟨.program ⟨257⟩, ⟨9595⟩⟩
def mergeEvent : Nat := 16391
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16387RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16387 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16387) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7254⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16391

namespace LeftMerge16396
def owner : Owner := ⟨.program ⟨257⟩, ⟨9656⟩⟩
def mergeEvent : Nat := 16396
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16392RawTerms
def rightRaw : List Term := Proof.Events064.exact16384RawTerms
def group : MergeGroup := .operator 16392 16384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16392) (leftOrdinal := 0)
    (rightResult := 16384) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9511⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16396

namespace LeftMerge16401
def owner : Owner := ⟨.program ⟨257⟩, ⟨9675⟩⟩
def mergeEvent : Nat := 16401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩, ⟨.program ⟨257⟩, ⟨7127⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16397RawTerms
def rightRaw : List Term := Proof.Events063.exact16374RawTerms
def group : MergeGroup := .operator 16397 16374
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16397) (leftOrdinal := 0)
    (rightResult := 16374) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7127⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩, ⟨.program ⟨257⟩, ⟨7127⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16401

namespace LeftMerge16406
def owner : Owner := ⟨.program ⟨257⟩, ⟨7040⟩⟩
def mergeEvent : Nat := 16406
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events032.exact8309RawTerms
def group : MergeGroup := .operator 2 8309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 8309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6806⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6806⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16406

namespace LeftMerge16431
def owner : Owner := ⟨.program ⟨257⟩, ⟨9596⟩⟩
def mergeEvent : Nat := 16431
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7256⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16427RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16427 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16427) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7256⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7256⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16431

namespace LeftMerge16436
def owner : Owner := ⟨.program ⟨257⟩, ⟨9657⟩⟩
def mergeEvent : Nat := 16436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7256⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9513⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16432RawTerms
def rightRaw : List Term := Proof.Events064.exact16424RawTerms
def group : MergeGroup := .operator 16432 16424
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16432) (leftOrdinal := 0)
    (rightResult := 16424) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7256⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9513⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7256⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9513⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16436

namespace LeftMerge16441
def owner : Owner := ⟨.program ⟨257⟩, ⟨9676⟩⟩
def mergeEvent : Nat := 16441
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7256⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9513⟩⟩, ⟨.program ⟨257⟩, ⟨7149⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16437RawTerms
def rightRaw : List Term := Proof.Events064.exact16414RawTerms
def group : MergeGroup := .operator 16437 16414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16437) (leftOrdinal := 0)
    (rightResult := 16414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7256⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7149⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7256⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9513⟩⟩, ⟨.program ⟨257⟩, ⟨7149⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16441

namespace LeftMerge16446
def owner : Owner := ⟨.program ⟨257⟩, ⟨7053⟩⟩
def mergeEvent : Nat := 16446
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events035.exact9057RawTerms
def group : MergeGroup := .operator 2 9057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 9057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6907⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6907⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16446

namespace LeftMerge16471
def owner : Owner := ⟨.program ⟨257⟩, ⟨9597⟩⟩
def mergeEvent : Nat := 16471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16467RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16467 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16467) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7258⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16471

namespace LeftMerge16476
def owner : Owner := ⟨.program ⟨257⟩, ⟨9658⟩⟩
def mergeEvent : Nat := 16476
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9515⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16472RawTerms
def rightRaw : List Term := Proof.Events064.exact16464RawTerms
def group : MergeGroup := .operator 16472 16464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16472) (leftOrdinal := 0)
    (rightResult := 16464) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9515⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9515⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16476

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
