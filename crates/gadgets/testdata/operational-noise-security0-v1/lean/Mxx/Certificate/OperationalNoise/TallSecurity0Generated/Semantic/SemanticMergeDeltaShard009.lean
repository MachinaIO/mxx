import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge3061
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 18)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3061

namespace LeftMerge3062
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 0)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3062

namespace LeftMerge3063
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 1)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3063

namespace LeftMerge3064
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3064
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 2)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3064

namespace LeftMerge3065
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3065
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 3)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3065

namespace LeftMerge3066
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3066
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 4)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3066

namespace LeftMerge3067
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3067
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 6)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3067

namespace LeftMerge3068
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3068
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 10)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3068

namespace LeftMerge3069
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3069
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 14)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3069

namespace LeftMerge3070
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3070
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 17)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3070

namespace LeftMerge3575
def owner : Owner := ⟨.program ⟨214⟩, ⟨18492⟩⟩
def mergeEvent : Nat := 3575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events013.exact3571RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 3571 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3571) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3575

namespace LeftMerge3583
def owner : Owner := ⟨.program ⟨214⟩, ⟨18121⟩⟩
def mergeEvent : Nat := 3583
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events013.exact3579RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 3579 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3579) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18120⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3583

namespace LeftMerge3591
def owner : Owner := ⟨.program ⟨214⟩, ⟨16924⟩⟩
def mergeEvent : Nat := 3591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3587RawTerms
def rightRaw : List Term := Proof.Events002.exact553RawTerms
def group : MergeGroup := .operator 3587 553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3587) (leftOrdinal := 0)
    (rightResult := 553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16923⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3591

namespace LeftMerge3599
def owner : Owner := ⟨.program ⟨214⟩, ⟨17491⟩⟩
def mergeEvent : Nat := 3599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3595RawTerms
def rightRaw : List Term := Proof.Events002.exact563RawTerms
def group : MergeGroup := .operator 3595 563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3595) (leftOrdinal := 0)
    (rightResult := 563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17490⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3599

namespace LeftMerge3607
def owner : Owner := ⟨.program ⟨214⟩, ⟨17715⟩⟩
def mergeEvent : Nat := 3607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3603RawTerms
def rightRaw : List Term := Proof.Events002.exact573RawTerms
def group : MergeGroup := .operator 3603 573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3603) (leftOrdinal := 0)
    (rightResult := 573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17714⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3607

namespace LeftMerge3615
def owner : Owner := ⟨.program ⟨214⟩, ⟨17946⟩⟩
def mergeEvent : Nat := 3615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3611RawTerms
def rightRaw : List Term := Proof.Events002.exact583RawTerms
def group : MergeGroup := .operator 3611 583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3611) (leftOrdinal := 0)
    (rightResult := 583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17945⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3615

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
