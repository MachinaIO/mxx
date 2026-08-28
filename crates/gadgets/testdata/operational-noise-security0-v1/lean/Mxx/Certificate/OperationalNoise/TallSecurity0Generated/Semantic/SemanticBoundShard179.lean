import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard178

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound27151
def owner : Owner := ⟨.program ⟨214⟩, ⟨15911⟩⟩
def transferEvent : Nat := 27151
def frameStart : Nat := 27078
def rule : BoundRule := .sum [.predecessor 0 27149 .coefficient, .predecessor 1 27150 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27149 .coefficient)
      LeftAuthority27147.bound (LeftAuthority27147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27147.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27150 .coefficient)
      LeftBound27143.bound (LeftBound27143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27143.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27147.bound, LeftBound27143.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27147.bound, LeftBound27143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority27147.actual selector witness, LeftBound27143.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27151

namespace LeftBound27155
def owner : Owner := ⟨.program ⟨214⟩, ⟨27689⟩⟩
def transferEvent : Nat := 27155
def frameStart : Nat := 27078
def rule : BoundRule := .product (.predecessor 0 27153 .coefficient) (.predecessor 1 27154 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27153 .coefficient)
      LeftBound27151.bound (LeftBound27151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27154 .coefficient)
      LeftAuthority27128.bound (LeftAuthority27128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27128.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27151.bound LeftAuthority27128.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27151.bound, LeftAuthority27128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27151.actual selector witness) * (LeftAuthority27128.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27155

namespace LeftBound27166
def owner : Owner := ⟨.program ⟨214⟩, ⟨15877⟩⟩
def transferEvent : Nat := 27166
def frameStart : Nat := 27078
def rule : BoundRule := .product (.predecessor 0 27164 .coefficient) (.predecessor 1 27165 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27164 .coefficient)
      LeftAuthority27139.bound (LeftAuthority27139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27139.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27165 .coefficient)
      LeftAuthority27162.bound (LeftAuthority27162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27162.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27162.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority27139.bound LeftAuthority27162.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27139.bound, LeftAuthority27162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority27139.actual selector witness) * (LeftAuthority27162.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27166

namespace LeftBound27174
def owner : Owner := ⟨.program ⟨214⟩, ⟨15878⟩⟩
def transferEvent : Nat := 27174
def frameStart : Nat := 27078
def rule : BoundRule := .sum [.predecessor 0 27172 .coefficient, .predecessor 1 27173 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27172 .coefficient)
      LeftAuthority27170.bound (LeftAuthority27170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27170.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27173 .coefficient)
      LeftBound27166.bound (LeftBound27166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27170.bound, LeftBound27166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27170.bound, LeftBound27166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority27170.actual selector witness, LeftBound27166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27174

namespace LeftBound27178
def owner : Owner := ⟨.program ⟨214⟩, ⟨27693⟩⟩
def transferEvent : Nat := 27178
def frameStart : Nat := 27078
def rule : BoundRule := .sum [.predecessor 0 27176 .coefficient, .predecessor 1 27177 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27176 .coefficient)
      LeftBound27174.bound (LeftBound27174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27177 .coefficient)
      LeftBound27155.bound (LeftBound27155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27155.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27174.bound, LeftBound27155.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27174.bound, LeftBound27155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27174.actual selector witness, LeftBound27155.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27178

namespace LeftBound27191
def owner : Owner := ⟨.program ⟨214⟩, ⟨27691⟩⟩
def transferEvent : Nat := 27191
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 27189 .coefficient, .predecessor 1 27190 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27189 .coefficient)
      LeftBound27020.bound (LeftBound27020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27190 .coefficient)
      LeftBound27003.bound (LeftBound27003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27020.bound, LeftBound27003.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27020.bound, LeftBound27003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27020.actual selector witness, LeftBound27003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27191

namespace LeftBound27194
def owner : Owner := ⟨.program ⟨214⟩, ⟨27691⟩⟩
def transferEvent : Nat := 27194
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 27188 .summary, .result 27010 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27188 .summary)
      LeftBound27022.bound (LeftBound27022.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21271⟩⟩) (rawTerms := some (Proof.Events106.exact27188RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27010 .summary)
      LeftBound27005.bound (LeftBound27005.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27690⟩⟩) (rawTerms := some (Proof.Events105.exact27010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27022.bound, LeftBound27005.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27022.bound, LeftBound27005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27022.actual selector witness, LeftBound27005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27194

namespace LeftBound27218
def owner : Owner := ⟨.program ⟨214⟩, ⟨11314⟩⟩
def transferEvent : Nat := 27218
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 27216 .coefficient) (.predecessor 1 27217 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27216 .coefficient)
      LeftAuthority1117.bound (LeftAuthority1117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1117.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27217 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1117.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1117.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1117.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27218

namespace LeftBound27223
def owner : Owner := ⟨.program ⟨214⟩, ⟨7347⟩⟩
def transferEvent : Nat := 27223
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27221 .coefficient) (.predecessor 1 27222 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27221 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27222 .coefficient)
      LeftBound12483.bound (LeftBound12483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12483.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound12483.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound12483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound12483.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27223

namespace LeftBound27228
def owner : Owner := ⟨.program ⟨214⟩, ⟨11315⟩⟩
def transferEvent : Nat := 27228
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 27226 .coefficient, .predecessor 1 27227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27226 .coefficient)
      LeftBound27223.bound (LeftBound27223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27227 .coefficient)
      LeftBound27218.bound (LeftBound27218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27218.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27223.bound, LeftBound27218.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27223.bound, LeftBound27218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27223.actual selector witness, LeftBound27218.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27228

namespace LeftBound27232
def owner : Owner := ⟨.program ⟨214⟩, ⟨11316⟩⟩
def transferEvent : Nat := 27232
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 27230 .coefficient, .predecessor 1 27231 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27230 .coefficient)
      LeftBound27228.bound (LeftBound27228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27231 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27228.bound, LeftBound12475.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27228.bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27228.actual selector witness, LeftBound12475.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27232

namespace LeftBound27233
def owner : Owner := ⟨.program ⟨214⟩, ⟨11316⟩⟩
def transferEvent : Nat := 27233
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩ [⟨.result 12476 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12476 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨91⟩⟩) (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12475.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12475.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27233

namespace LeftBound27238
def owner : Owner := ⟨.program ⟨214⟩, ⟨13803⟩⟩
def transferEvent : Nat := 27238
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27236 .coefficient) (.predecessor 1 27237 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27236 .coefficient)
      LeftBound27232.bound (LeftBound27232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27232.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27237 .coefficient)
      LeftAuthority1120.bound (LeftAuthority1120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound27232.bound LeftAuthority1120.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27232.bound, LeftAuthority1120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound27232.actual selector witness) * (LeftAuthority1120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27238

namespace LeftBound27239
def owner : Owner := ⟨.program ⟨214⟩, ⟨13803⟩⟩
def transferEvent : Nat := 27239
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13800⟩⟩], []⟩ [⟨.result 1121 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1121 .coefficient)
      LeftAuthority1120.bound (LeftAuthority1120.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13800⟩⟩) (rawTerms := some (Proof.Events004.exact1121RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1120.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1120.bound []
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1120.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27239

namespace LeftBound27240
def owner : Owner := ⟨.program ⟨214⟩, ⟨13803⟩⟩
def transferEvent : Nat := 27240
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 27235 .summary) (.transfer 27239) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27235 .summary)
      LeftBound27233.bound (LeftBound27233.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11316⟩⟩) (rawTerms := some (Proof.Events106.exact27235RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27239)
      LeftBound27239.bound (LeftBound27239.actual selector witness) := by
  exact .transfer (LeftBound27239.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound27233.bound LeftBound27239.bound
def bound : CoeffClass := .finite ⟨9984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27233.bound, LeftBound27239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound27233.actual selector witness) * (LeftBound27239.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27240

namespace LeftBound27246
def owner : Owner := ⟨.program ⟨214⟩, ⟨13804⟩⟩
def transferEvent : Nat := 27246
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 27244 .coefficient) (.predecessor 1 27245 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27244 .coefficient)
      LeftAuthority1120.bound (LeftAuthority1120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1120.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27245 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1120.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1120.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1120.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27246

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
