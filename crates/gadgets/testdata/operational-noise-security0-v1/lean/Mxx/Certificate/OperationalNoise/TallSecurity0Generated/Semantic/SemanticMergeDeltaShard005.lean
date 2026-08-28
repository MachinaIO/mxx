import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge2095
def owner : Owner := ⟨.program ⟨214⟩, ⟨16936⟩⟩
def mergeEvent : Nat := 2095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2091RawTerms
def rightRaw : List Term := Proof.Events002.exact553RawTerms
def group : MergeGroup := .operator 2091 553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2091) (leftOrdinal := 0)
    (rightResult := 553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16935⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2095

namespace LeftMerge2103
def owner : Owner := ⟨.program ⟨214⟩, ⟨17503⟩⟩
def mergeEvent : Nat := 2103
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2099RawTerms
def rightRaw : List Term := Proof.Events002.exact563RawTerms
def group : MergeGroup := .operator 2099 563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2099) (leftOrdinal := 0)
    (rightResult := 563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17502⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2103

namespace LeftMerge2111
def owner : Owner := ⟨.program ⟨214⟩, ⟨17727⟩⟩
def mergeEvent : Nat := 2111
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2107RawTerms
def rightRaw : List Term := Proof.Events002.exact573RawTerms
def group : MergeGroup := .operator 2107 573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2107) (leftOrdinal := 0)
    (rightResult := 573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17726⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2111

namespace LeftMerge2119
def owner : Owner := ⟨.program ⟨214⟩, ⟨17958⟩⟩
def mergeEvent : Nat := 2119
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2115RawTerms
def rightRaw : List Term := Proof.Events002.exact583RawTerms
def group : MergeGroup := .operator 2115 583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2115) (leftOrdinal := 0)
    (rightResult := 583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17957⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2119

namespace LeftMerge2127
def owner : Owner := ⟨.program ⟨214⟩, ⟨17559⟩⟩
def mergeEvent : Nat := 2127
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2123RawTerms
def rightRaw : List Term := Proof.Events002.exact593RawTerms
def group : MergeGroup := .operator 2123 593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2123) (leftOrdinal := 0)
    (rightResult := 593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17558⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2127

namespace LeftMerge2135
def owner : Owner := ⟨.program ⟨214⟩, ⟨18864⟩⟩
def mergeEvent : Nat := 2135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2131RawTerms
def rightRaw : List Term := Proof.Events002.exact603RawTerms
def group : MergeGroup := .operator 2131 603
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2131) (leftOrdinal := 0)
    (rightResult := 603) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18863⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2135

namespace LeftMerge2143
def owner : Owner := ⟨.program ⟨214⟩, ⟨17615⟩⟩
def mergeEvent : Nat := 2143
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2139RawTerms
def rightRaw : List Term := Proof.Events002.exact613RawTerms
def group : MergeGroup := .operator 2139 613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2139) (leftOrdinal := 0)
    (rightResult := 613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2143

namespace LeftMerge2151
def owner : Owner := ⟨.program ⟨214⟩, ⟨17671⟩⟩
def mergeEvent : Nat := 2151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2147RawTerms
def rightRaw : List Term := Proof.Events002.exact623RawTerms
def group : MergeGroup := .operator 2147 623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2147) (leftOrdinal := 0)
    (rightResult := 623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17670⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2151

namespace LeftMerge2159
def owner : Owner := ⟨.program ⟨214⟩, ⟨18050⟩⟩
def mergeEvent : Nat := 2159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2155RawTerms
def rightRaw : List Term := Proof.Events002.exact633RawTerms
def group : MergeGroup := .operator 2155 633
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2155) (leftOrdinal := 0)
    (rightResult := 633) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18049⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2159

namespace LeftMerge2167
def owner : Owner := ⟨.program ⟨214⟩, ⟨17174⟩⟩
def mergeEvent : Nat := 2167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2163RawTerms
def rightRaw : List Term := Proof.Events002.exact643RawTerms
def group : MergeGroup := .operator 2163 643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2163) (leftOrdinal := 0)
    (rightResult := 643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2167

namespace LeftMerge2175
def owner : Owner := ⟨.program ⟨214⟩, ⟨17230⟩⟩
def mergeEvent : Nat := 2175
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2171RawTerms
def rightRaw : List Term := Proof.Events002.exact653RawTerms
def group : MergeGroup := .operator 2171 653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2171) (leftOrdinal := 0)
    (rightResult := 653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2175

namespace LeftMerge2183
def owner : Owner := ⟨.program ⟨214⟩, ⟨17447⟩⟩
def mergeEvent : Nat := 2183
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2179RawTerms
def rightRaw : List Term := Proof.Events002.exact663RawTerms
def group : MergeGroup := .operator 2179 663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2179) (leftOrdinal := 0)
    (rightResult := 663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2183

namespace LeftMerge2191
def owner : Owner := ⟨.program ⟨214⟩, ⟨17831⟩⟩
def mergeEvent : Nat := 2191
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2187RawTerms
def rightRaw : List Term := Proof.Events002.exact673RawTerms
def group : MergeGroup := .operator 2187 673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2187) (leftOrdinal := 0)
    (rightResult := 673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2191

namespace LeftMerge2199
def owner : Owner := ⟨.program ⟨214⟩, ⟨15527⟩⟩
def mergeEvent : Nat := 2199
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2195RawTerms
def rightRaw : List Term := Proof.Events002.exact683RawTerms
def group : MergeGroup := .operator 2195 683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2195) (leftOrdinal := 0)
    (rightResult := 683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2199

namespace LeftMerge2207
def owner : Owner := ⟨.program ⟨214⟩, ⟨15219⟩⟩
def mergeEvent : Nat := 2207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2203RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 2203 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2203) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2207

namespace LeftMerge2215
def owner : Owner := ⟨.program ⟨214⟩, ⟨15058⟩⟩
def mergeEvent : Nat := 2215
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2211RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 2211 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2211) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2215

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
