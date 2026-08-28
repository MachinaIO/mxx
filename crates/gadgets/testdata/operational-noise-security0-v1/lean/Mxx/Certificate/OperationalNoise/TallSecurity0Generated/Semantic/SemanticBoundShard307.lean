import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard304
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard305
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard306

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound46116
def owner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def transferEvent : Nat := 46116
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46114 .coefficient, .predecessor 1 46115 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46114 .coefficient)
      LeftBound46112.bound (LeftBound46112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46115 .coefficient)
      LeftAuthority46053.bound (LeftAuthority46053.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46053.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46053.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46112.bound, LeftAuthority46053.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46112.bound, LeftAuthority46053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46112.actual selector witness, LeftAuthority46053.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46116

namespace LeftBound46120
def owner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def transferEvent : Nat := 46120
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46118 .coefficient, .predecessor 1 46119 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46118 .coefficient)
      LeftBound46116.bound (LeftBound46116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46119 .coefficient)
      LeftAuthority46050.bound (LeftAuthority46050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46050.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46116.bound, LeftAuthority46050.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46116.bound, LeftAuthority46050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46116.actual selector witness, LeftAuthority46050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46120

namespace LeftBound46124
def owner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def transferEvent : Nat := 46124
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46122 .coefficient, .predecessor 1 46123 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46122 .coefficient)
      LeftBound46120.bound (LeftBound46120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46120.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46123 .coefficient)
      LeftAuthority46047.bound (LeftAuthority46047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46047.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46047.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46120.bound, LeftAuthority46047.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46120.bound, LeftAuthority46047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46120.actual selector witness, LeftAuthority46047.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46124

namespace LeftBound46128
def owner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def transferEvent : Nat := 46128
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46126 .coefficient, .predecessor 1 46127 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46126 .coefficient)
      LeftBound46124.bound (LeftBound46124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46127 .coefficient)
      LeftAuthority46044.bound (LeftAuthority46044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46044.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46044.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46124.bound, LeftAuthority46044.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46124.bound, LeftAuthority46044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46124.actual selector witness, LeftAuthority46044.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46128

namespace LeftBound46132
def owner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def transferEvent : Nat := 46132
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46130 .coefficient, .predecessor 1 46131 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46130 .coefficient)
      LeftBound46128.bound (LeftBound46128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46131 .coefficient)
      LeftAuthority46041.bound (LeftAuthority46041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46041.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46041.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46128.bound, LeftAuthority46041.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46128.bound, LeftAuthority46041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46128.actual selector witness, LeftAuthority46041.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46132

namespace LeftBound46136
def owner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def transferEvent : Nat := 46136
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46134 .coefficient, .predecessor 1 46135 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46134 .coefficient)
      LeftBound46132.bound (LeftBound46132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46132.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46135 .coefficient)
      LeftAuthority46038.bound (LeftAuthority46038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46038.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46132.bound, LeftAuthority46038.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46132.bound, LeftAuthority46038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46132.actual selector witness, LeftAuthority46038.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46136

namespace LeftBound46140
def owner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def transferEvent : Nat := 46140
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46138 .coefficient, .predecessor 1 46139 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46138 .coefficient)
      LeftBound46136.bound (LeftBound46136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46139 .coefficient)
      LeftAuthority46035.bound (LeftAuthority46035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46035.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46136.bound, LeftAuthority46035.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46136.bound, LeftAuthority46035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46136.actual selector witness, LeftAuthority46035.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46140

namespace LeftBound46144
def owner : Owner := ⟨.program ⟨214⟩, ⟨6810⟩⟩
def transferEvent : Nat := 46144
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46142 .coefficient, .predecessor 1 46143 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46142 .coefficient)
      LeftBound46140.bound (LeftBound46140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46143 .coefficient)
      LeftAuthority46032.bound (LeftAuthority46032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46032.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46140.bound, LeftAuthority46032.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46140.bound, LeftAuthority46032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46140.actual selector witness, LeftAuthority46032.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46144

namespace LeftBound46148
def owner : Owner := ⟨.program ⟨214⟩, ⟨6811⟩⟩
def transferEvent : Nat := 46148
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46146 .coefficient, .predecessor 1 46147 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46146 .coefficient)
      LeftBound46144.bound (LeftBound46144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46144.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46147 .coefficient)
      LeftAuthority46029.bound (LeftAuthority46029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46029.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46144.bound, LeftAuthority46029.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46144.bound, LeftAuthority46029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46144.actual selector witness, LeftAuthority46029.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46148

namespace LeftBound46152
def owner : Owner := ⟨.program ⟨214⟩, ⟨18658⟩⟩
def transferEvent : Nat := 46152
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46150 .coefficient, .predecessor 1 46151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46150 .coefficient)
      LeftBound46148.bound (LeftBound46148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46148.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46151 .coefficient)
      LeftBound46008.bound (LeftBound46008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46008.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46148.bound, LeftBound46008.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46148.bound, LeftBound46008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46148.actual selector witness, LeftBound46008.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46152

namespace LeftBound46156
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def transferEvent : Nat := 46156
def frameStart : Nat := 45478
def rule : BoundRule := .product (.predecessor 0 46154 .coefficient) (.predecessor 1 46155 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46154 .coefficient)
      LeftBound46152.bound (LeftBound46152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46155 .coefficient)
      LeftAuthority45993.bound (LeftAuthority45993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45993.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45993.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46152.bound LeftAuthority45993.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46152.bound, LeftAuthority45993.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46152.actual selector witness) * (LeftAuthority45993.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46156

namespace LeftBound46235
def owner : Owner := ⟨.program ⟨214⟩, ⟨18505⟩⟩
def transferEvent : Nat := 46235
def frameStart : Nat := 45478
def rule : BoundRule := .product (.predecessor 0 46233 .coefficient) (.predecessor 1 46234 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46233 .coefficient)
      LeftAuthority46004.bound (LeftAuthority46004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46234 .coefficient)
      LeftAuthority46231.bound (LeftAuthority46231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46231.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority46004.bound LeftAuthority46231.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46004.bound, LeftAuthority46231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority46004.actual selector witness) * (LeftAuthority46231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46235

namespace LeftBound46243
def owner : Owner := ⟨.program ⟨214⟩, ⟨18506⟩⟩
def transferEvent : Nat := 46243
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46241 .coefficient, .predecessor 1 46242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46241 .coefficient)
      LeftAuthority46239.bound (LeftAuthority46239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46242 .coefficient)
      LeftBound46235.bound (LeftBound46235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46235.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority46239.bound, LeftBound46235.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46239.bound, LeftBound46235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority46239.actual selector witness, LeftBound46235.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46243

namespace LeftBound46247
def owner : Owner := ⟨.program ⟨214⟩, ⟨18689⟩⟩
def transferEvent : Nat := 46247
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46245 .coefficient, .predecessor 1 46246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46245 .coefficient)
      LeftBound46243.bound (LeftBound46243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46246 .coefficient)
      LeftBound46156.bound (LeftBound46156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46156.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46243.bound, LeftBound46156.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46243.bound, LeftBound46156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46243.actual selector witness, LeftBound46156.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46247

namespace LeftBound46294
def owner : Owner := ⟨.program ⟨214⟩, ⟨30167⟩⟩
def transferEvent : Nat := 46294
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46292 .coefficient, .predecessor 1 46293 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46292 .coefficient)
      LeftBound44885.bound (LeftBound44885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44885.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44885.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46293 .coefficient)
      LeftBound44800.bound (LeftBound44800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact44875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44885.bound, LeftBound44800.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44885.bound, LeftBound44800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44885.actual selector witness, LeftBound44800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46294

namespace LeftBound46331
def owner : Owner := ⟨.program ⟨214⟩, ⟨30167⟩⟩
def transferEvent : Nat := 46331
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46291 .summary, .result 44875 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46291 .summary)
      LeftBound44887.bound (LeftBound44887.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18570⟩⟩) (rawTerms := some (Proof.Events180.exact46291RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44875 .summary)
      LeftBound44802.bound (LeftBound44802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30166⟩⟩) (rawTerms := some (Proof.Events175.exact44875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44887.bound, LeftBound44802.bound]
def bound : CoeffClass := .finite ⟨85361036953731455419885957120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44887.bound, LeftBound44802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44887.actual selector witness, LeftBound44802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46331

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
