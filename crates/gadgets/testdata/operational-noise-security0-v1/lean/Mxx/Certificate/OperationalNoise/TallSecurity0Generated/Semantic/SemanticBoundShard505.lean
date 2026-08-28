import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard490
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard494
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard498
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard501
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard504

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound73910
def owner : Owner := ⟨.program ⟨214⟩, ⟨14830⟩⟩
def transferEvent : Nat := 73910
def frameStart : Nat := 73845
def rule : BoundRule := .product (.predecessor 0 73908 .coefficient) (.predecessor 1 73909 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73908 .coefficient)
      LeftAuthority73906.bound (LeftAuthority73906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73906.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73909 .coefficient)
      LeftBound73904.bound (LeftBound73904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73904.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority73906.bound LeftBound73904.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73906.bound, LeftBound73904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority73906.actual selector witness) * (LeftBound73904.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73910

namespace LeftBound73918
def owner : Owner := ⟨.program ⟨214⟩, ⟨14831⟩⟩
def transferEvent : Nat := 73918
def frameStart : Nat := 73845
def rule : BoundRule := .sum [.predecessor 0 73916 .coefficient, .predecessor 1 73917 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73916 .coefficient)
      LeftAuthority73914.bound (LeftAuthority73914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73917 .coefficient)
      LeftBound73910.bound (LeftBound73910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73910.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority73914.bound, LeftBound73910.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73914.bound, LeftBound73910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority73914.actual selector witness, LeftBound73910.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73918

namespace LeftBound73922
def owner : Owner := ⟨.program ⟨214⟩, ⟨26347⟩⟩
def transferEvent : Nat := 73922
def frameStart : Nat := 73845
def rule : BoundRule := .product (.predecessor 0 73920 .coefficient) (.predecessor 1 73921 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73920 .coefficient)
      LeftBound73918.bound (LeftBound73918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73918.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73921 .coefficient)
      LeftAuthority73895.bound (LeftAuthority73895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73895.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73895.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73918.bound LeftAuthority73895.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73918.bound, LeftAuthority73895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73918.actual selector witness) * (LeftAuthority73895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73922

namespace LeftBound73933
def owner : Owner := ⟨.program ⟨214⟩, ⟨15263⟩⟩
def transferEvent : Nat := 73933
def frameStart : Nat := 73845
def rule : BoundRule := .product (.predecessor 0 73931 .coefficient) (.predecessor 1 73932 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73931 .coefficient)
      LeftAuthority73906.bound (LeftAuthority73906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73906.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73932 .coefficient)
      LeftAuthority73929.bound (LeftAuthority73929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73929.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority73906.bound LeftAuthority73929.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73906.bound, LeftAuthority73929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority73906.actual selector witness) * (LeftAuthority73929.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73933

namespace LeftBound73941
def owner : Owner := ⟨.program ⟨214⟩, ⟨15264⟩⟩
def transferEvent : Nat := 73941
def frameStart : Nat := 73845
def rule : BoundRule := .sum [.predecessor 0 73939 .coefficient, .predecessor 1 73940 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73939 .coefficient)
      LeftAuthority73937.bound (LeftAuthority73937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73940 .coefficient)
      LeftBound73933.bound (LeftBound73933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73933.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73933.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority73937.bound, LeftBound73933.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73937.bound, LeftBound73933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority73937.actual selector witness, LeftBound73933.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73941

namespace LeftBound73945
def owner : Owner := ⟨.program ⟨214⟩, ⟨26350⟩⟩
def transferEvent : Nat := 73945
def frameStart : Nat := 73845
def rule : BoundRule := .sum [.predecessor 0 73943 .coefficient, .predecessor 1 73944 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73943 .coefficient)
      LeftBound73941.bound (LeftBound73941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73941.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73944 .coefficient)
      LeftBound73922.bound (LeftBound73922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73941.bound, LeftBound73922.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73941.bound, LeftBound73922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73941.actual selector witness, LeftBound73922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73945

namespace LeftBound73958
def owner : Owner := ⟨.program ⟨214⟩, ⟨26349⟩⟩
def transferEvent : Nat := 73958
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73956 .coefficient, .predecessor 1 73957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73956 .coefficient)
      LeftBound73787.bound (LeftBound73787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73787.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73957 .coefficient)
      LeftBound73770.bound (LeftBound73770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73770.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73787.bound, LeftBound73770.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73787.bound, LeftBound73770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73787.actual selector witness, LeftBound73770.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73958

namespace LeftBound73961
def owner : Owner := ⟨.program ⟨214⟩, ⟨26349⟩⟩
def transferEvent : Nat := 73961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73955 .summary, .result 73777 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73955 .summary)
      LeftBound73789.bound (LeftBound73789.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20391⟩⟩) (rawTerms := some (Proof.Events288.exact73955RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73777 .summary)
      LeftBound73772.bound (LeftBound73772.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26348⟩⟩) (rawTerms := some (Proof.Events288.exact73777RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73772.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73789.bound, LeftBound73772.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73789.bound, LeftBound73772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73789.actual selector witness, LeftBound73772.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73961

namespace LeftBound73965
def owner : Owner := ⟨.program ⟨214⟩, ⟨26555⟩⟩
def transferEvent : Nat := 73965
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73963 .coefficient, .predecessor 1 73964 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73963 .coefficient)
      LeftBound73958.bound (LeftBound73958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73964 .coefficient)
      LeftBound73476.bound (LeftBound73476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73476.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73958.bound, LeftBound73476.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73958.bound, LeftBound73476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73958.actual selector witness, LeftBound73476.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73965

namespace LeftBound73966
def owner : Owner := ⟨.program ⟨214⟩, ⟨26555⟩⟩
def transferEvent : Nat := 73966
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73962 .summary, .result 73480 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73962 .summary)
      LeftBound73961.bound (LeftBound73961.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26349⟩⟩) (rawTerms := some (Proof.Events288.exact73962RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73480 .summary)
      LeftBound73479.bound (LeftBound73479.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26554⟩⟩) (rawTerms := some (Proof.Events287.exact73480RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73961.bound, LeftBound73479.bound]
def bound : CoeffClass := .finite ⟨2583789554981353578496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73961.bound, LeftBound73479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73961.actual selector witness, LeftBound73479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73966

namespace LeftBound73970
def owner : Owner := ⟨.program ⟨214⟩, ⟨26772⟩⟩
def transferEvent : Nat := 73970
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73968 .coefficient, .predecessor 1 73969 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73968 .coefficient)
      LeftBound73965.bound (LeftBound73965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73965.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73965.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73969 .coefficient)
      LeftBound72994.bound (LeftBound72994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact72998RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72994.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73965.bound, LeftBound72994.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73965.bound, LeftBound72994.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73965.actual selector witness, LeftBound72994.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73970

namespace LeftBound73971
def owner : Owner := ⟨.program ⟨214⟩, ⟨26772⟩⟩
def transferEvent : Nat := 73971
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73967 .summary, .result 72998 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73967 .summary)
      LeftBound73966.bound (LeftBound73966.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26555⟩⟩) (rawTerms := some (Proof.Events288.exact73967RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72998 .summary)
      LeftBound72997.bound (LeftBound72997.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26771⟩⟩) (rawTerms := some (Proof.Events285.exact72998RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72997.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73966.bound, LeftBound72997.bound]
def bound : CoeffClass := .finite ⟨3875701141805795807232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73966.bound, LeftBound72997.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73966.actual selector witness, LeftBound72997.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73971

namespace LeftBound73975
def owner : Owner := ⟨.program ⟨214⟩, ⟨26989⟩⟩
def transferEvent : Nat := 73975
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73973 .coefficient, .predecessor 1 73974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73973 .coefficient)
      LeftBound73970.bound (LeftBound73970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73970.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73974 .coefficient)
      LeftBound72512.bound (LeftBound72512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73970.bound, LeftBound72512.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73970.bound, LeftBound72512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73970.actual selector witness, LeftBound72512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73975

namespace LeftBound73976
def owner : Owner := ⟨.program ⟨214⟩, ⟨26989⟩⟩
def transferEvent : Nat := 73976
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73972 .summary, .result 72516 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73972 .summary)
      LeftBound73971.bound (LeftBound73971.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26772⟩⟩) (rawTerms := some (Proof.Events288.exact73972RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72516 .summary)
      LeftBound72515.bound (LeftBound72515.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26988⟩⟩) (rawTerms := some (Proof.Events283.exact72516RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73971.bound, LeftBound72515.bound]
def bound : CoeffClass := .finite ⟨5167635141075258621952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73971.bound, LeftBound72515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73971.actual selector witness, LeftBound72515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73976

namespace LeftBound73980
def owner : Owner := ⟨.program ⟨214⟩, ⟨27206⟩⟩
def transferEvent : Nat := 73980
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73978 .coefficient, .predecessor 1 73979 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73978 .coefficient)
      LeftBound73975.bound (LeftBound73975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73979 .coefficient)
      LeftBound72030.bound (LeftBound72030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72030.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73975.bound, LeftBound72030.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73975.bound, LeftBound72030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73975.actual selector witness, LeftBound72030.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73980

namespace LeftBound73981
def owner : Owner := ⟨.program ⟨214⟩, ⟨27206⟩⟩
def transferEvent : Nat := 73981
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73977 .summary, .result 72034 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73977 .summary)
      LeftBound73976.bound (LeftBound73976.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26989⟩⟩) (rawTerms := some (Proof.Events288.exact73977RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72034 .summary)
      LeftBound72033.bound (LeftBound72033.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27205⟩⟩) (rawTerms := some (Proof.Events281.exact72034RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72033.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73976.bound, LeftBound72033.bound]
def bound : CoeffClass := .finite ⟨6459613965234762608640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73976.bound, LeftBound72033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73976.actual selector witness, LeftBound72033.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73981

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
