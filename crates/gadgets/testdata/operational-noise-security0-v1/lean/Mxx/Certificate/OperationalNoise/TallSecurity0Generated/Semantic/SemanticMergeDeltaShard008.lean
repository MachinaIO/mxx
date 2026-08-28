import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge2923
def owner : Owner := ⟨.program ⟨214⟩, ⟨17226⟩⟩
def mergeEvent : Nat := 2923
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact2919RawTerms
def rightRaw : List Term := Proof.Events002.exact653RawTerms
def group : MergeGroup := .operator 2919 653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2919) (leftOrdinal := 0)
    (rightResult := 653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17225⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2923

namespace LeftMerge2931
def owner : Owner := ⟨.program ⟨214⟩, ⟨17443⟩⟩
def mergeEvent : Nat := 2931
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact2927RawTerms
def rightRaw : List Term := Proof.Events002.exact663RawTerms
def group : MergeGroup := .operator 2927 663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2927) (leftOrdinal := 0)
    (rightResult := 663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17442⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2931

namespace LeftMerge2939
def owner : Owner := ⟨.program ⟨214⟩, ⟨17823⟩⟩
def mergeEvent : Nat := 2939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact2935RawTerms
def rightRaw : List Term := Proof.Events002.exact673RawTerms
def group : MergeGroup := .operator 2935 673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2935) (leftOrdinal := 0)
    (rightResult := 673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17822⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2939

namespace LeftMerge2947
def owner : Owner := ⟨.program ⟨214⟩, ⟨15522⟩⟩
def mergeEvent : Nat := 2947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact2943RawTerms
def rightRaw : List Term := Proof.Events002.exact683RawTerms
def group : MergeGroup := .operator 2943 683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2943) (leftOrdinal := 0)
    (rightResult := 683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15521⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2947

namespace LeftMerge2955
def owner : Owner := ⟨.program ⟨214⟩, ⟨15214⟩⟩
def mergeEvent : Nat := 2955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact2951RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 2951 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2951) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15213⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2955

namespace LeftMerge2963
def owner : Owner := ⟨.program ⟨214⟩, ⟨15053⟩⟩
def mergeEvent : Nat := 2963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact2959RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 2959 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2959) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15052⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2963

namespace LeftMerge2971
def owner : Owner := ⟨.program ⟨214⟩, ⟨14892⟩⟩
def mergeEvent : Nat := 2971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact2967RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 2967 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2967) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14891⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2971

namespace LeftMerge3052
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 5)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge3052

namespace LeftMerge3053
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 7)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3053

namespace LeftMerge3054
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3054
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 8)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3054

namespace LeftMerge3055
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 9)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3055

namespace LeftMerge3056
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 11)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3056

namespace LeftMerge3057
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 12)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3057

namespace LeftMerge3058
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3058
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 13)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3058

namespace LeftMerge3059
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 15)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3059

namespace LeftMerge3060
def owner : Owner := ⟨.program ⟨214⟩, ⟨18859⟩⟩
def mergeEvent : Nat := 3060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 16)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6493⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3060

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
