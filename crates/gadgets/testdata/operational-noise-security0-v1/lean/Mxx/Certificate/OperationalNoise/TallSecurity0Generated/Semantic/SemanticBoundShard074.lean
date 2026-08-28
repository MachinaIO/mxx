import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard073

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound12156
def owner : Owner := ⟨.program ⟨214⟩, ⟨14027⟩⟩
def transferEvent : Nat := 12156
def frameStart : Nat := 12127
def rule : BoundRule := .product (.predecessor 0 12154 .coefficient) (.predecessor 1 12155 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12154 .coefficient)
      LeftAuthority12152.bound (LeftAuthority12152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12155 .coefficient)
      LeftAuthority12149.bound (LeftAuthority12149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12149.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12149.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority12152.bound LeftAuthority12149.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12152.bound, LeftAuthority12149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority12152.actual selector witness) * (LeftAuthority12149.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12156

namespace LeftBound12160
def owner : Owner := ⟨.program ⟨214⟩, ⟨14028⟩⟩
def transferEvent : Nat := 12160
def frameStart : Nat := 12127
def rule : BoundRule := .identity (.predecessor 0 12159 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12159 .coefficient)
      LeftBound12156.bound (LeftBound12156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12156.derived selector witness)

def rawBound : CoeffClass := LeftBound12156.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound12156.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12160

namespace LeftBound12177
def owner : Owner := ⟨.program ⟨214⟩, ⟨14113⟩⟩
def transferEvent : Nat := 12177
def frameStart : Nat := 12127
def rule : BoundRule := .sum [.predecessor 0 12175 .coefficient, .predecessor 1 12176 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12175 .coefficient)
      LeftBound12160.bound (LeftBound12160.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound12160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12176 .coefficient)
      LeftAuthority12173.bound (LeftAuthority12173.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority12173.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12160.bound, LeftAuthority12173.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12160.bound, LeftAuthority12173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12160.actual selector witness, LeftAuthority12173.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12177

namespace LeftBound12180
def owner : Owner := ⟨.program ⟨214⟩, ⟨14114⟩⟩
def transferEvent : Nat := 12180
def frameStart : Nat := 12127
def rule : BoundRule := .identity (.predecessor 0 12179 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12179 .coefficient)
      LeftBound12177.bound (LeftBound12177.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound12177.derived selector witness)

def rawBound : CoeffClass := LeftBound12177.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound12177.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12180

namespace LeftBound12186
def owner : Owner := ⟨.program ⟨214⟩, ⟨14115⟩⟩
def transferEvent : Nat := 12186
def frameStart : Nat := 12127
def rule : BoundRule := .product (.predecessor 0 12184 .coefficient) (.predecessor 1 12185 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12184 .coefficient)
      LeftAuthority12182.bound (LeftAuthority12182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12185 .coefficient)
      LeftBound12180.bound (LeftBound12180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12180.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority12182.bound LeftBound12180.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12182.bound, LeftBound12180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority12182.actual selector witness) * (LeftBound12180.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12186

namespace LeftBound12202
def owner : Owner := ⟨.program ⟨214⟩, ⟨7850⟩⟩
def transferEvent : Nat := 12202
def frameStart : Nat := 12127
def rule : BoundRule := .scale (.predecessor 0 12200 .coefficient) (.value (.predecessor 1 12201 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12200 .coefficient)
      LeftAuthority12198.bound (LeftAuthority12198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12201 .coefficient)
      LeftAuthority12189.bound (LeftAuthority12189.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority12189.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority12198.bound LeftAuthority12189.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12198.bound, LeftAuthority12189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12198.actual selector witness) * (LeftAuthority12189.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound12202

namespace LeftBound12205
def owner : Owner := ⟨.program ⟨214⟩, ⟨6758⟩⟩
def transferEvent : Nat := 12205
def frameStart : Nat := 12127
def rule : BoundRule := .identity (.predecessor 0 12204 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12204 .coefficient)
      LeftAuthority12192.bound (LeftAuthority12192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12192.derived selector witness)

def rawBound : CoeffClass := LeftAuthority12192.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority12192.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12205

namespace LeftBound12209
def owner : Owner := ⟨.program ⟨214⟩, ⟨7851⟩⟩
def transferEvent : Nat := 12209
def frameStart : Nat := 12127
def rule : BoundRule := .product (.predecessor 0 12207 .coefficient) (.predecessor 1 12208 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12207 .coefficient)
      LeftBound12205.bound (LeftBound12205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12205.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12208 .coefficient)
      LeftBound12202.bound (LeftBound12202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12202.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12205.bound LeftBound12202.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12205.bound, LeftBound12202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12205.actual selector witness) * (LeftBound12202.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12209

namespace LeftBound12214
def owner : Owner := ⟨.program ⟨214⟩, ⟨14116⟩⟩
def transferEvent : Nat := 12214
def frameStart : Nat := 12127
def rule : BoundRule := .sum [.predecessor 0 12212 .coefficient, .predecessor 1 12213 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12212 .coefficient)
      LeftBound12209.bound (LeftBound12209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12213 .coefficient)
      LeftBound12186.bound (LeftBound12186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12186.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12209.bound, LeftBound12186.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12209.bound, LeftBound12186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12209.actual selector witness, LeftBound12186.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12214

namespace LeftBound12218
def owner : Owner := ⟨.program ⟨214⟩, ⟨26012⟩⟩
def transferEvent : Nat := 12218
def frameStart : Nat := 12127
def rule : BoundRule := .product (.predecessor 0 12216 .coefficient) (.predecessor 1 12217 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12216 .coefficient)
      LeftBound12214.bound (LeftBound12214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12217 .coefficient)
      LeftAuthority12171.bound (LeftAuthority12171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12171.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12171.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12214.bound LeftAuthority12171.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12214.bound, LeftAuthority12171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12214.actual selector witness) * (LeftAuthority12171.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12218

namespace LeftBound12229
def owner : Owner := ⟨.program ⟨214⟩, ⟨15839⟩⟩
def transferEvent : Nat := 12229
def frameStart : Nat := 12127
def rule : BoundRule := .product (.predecessor 0 12227 .coefficient) (.predecessor 1 12228 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12227 .coefficient)
      LeftAuthority12182.bound (LeftAuthority12182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12228 .coefficient)
      LeftAuthority12225.bound (LeftAuthority12225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12225.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12225.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority12182.bound LeftAuthority12225.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12182.bound, LeftAuthority12225.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority12182.actual selector witness) * (LeftAuthority12225.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12229

namespace LeftBound12237
def owner : Owner := ⟨.program ⟨214⟩, ⟨15840⟩⟩
def transferEvent : Nat := 12237
def frameStart : Nat := 12127
def rule : BoundRule := .sum [.predecessor 0 12235 .coefficient, .predecessor 1 12236 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12235 .coefficient)
      LeftAuthority12233.bound (LeftAuthority12233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12233.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12236 .coefficient)
      LeftBound12229.bound (LeftBound12229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12229.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority12233.bound, LeftBound12229.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12233.bound, LeftBound12229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority12233.actual selector witness, LeftBound12229.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12237

namespace LeftBound12241
def owner : Owner := ⟨.program ⟨214⟩, ⟨26013⟩⟩
def transferEvent : Nat := 12241
def frameStart : Nat := 12127
def rule : BoundRule := .sum [.predecessor 0 12239 .coefficient, .predecessor 1 12240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12239 .coefficient)
      LeftBound12237.bound (LeftBound12237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12240 .coefficient)
      LeftBound12218.bound (LeftBound12218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12218.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12237.bound, LeftBound12218.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12237.bound, LeftBound12218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12237.actual selector witness, LeftBound12218.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12241

namespace LeftBound12254
def owner : Owner := ⟨.program ⟨214⟩, ⟨26011⟩⟩
def transferEvent : Nat := 12254
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12252 .coefficient, .predecessor 1 12253 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12252 .coefficient)
      LeftBound12075.bound (LeftBound12075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12253 .coefficient)
      LeftBound12058.bound (LeftBound12058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12075.bound, LeftBound12058.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12075.bound, LeftBound12058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12075.actual selector witness, LeftBound12058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12254

namespace LeftBound12257
def owner : Owner := ⟨.program ⟨214⟩, ⟨26011⟩⟩
def transferEvent : Nat := 12257
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 12251 .summary, .result 12065 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12251 .summary)
      LeftBound12077.bound (LeftBound12077.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19475⟩⟩) (rawTerms := some (Proof.Events047.exact12251RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12077.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12065 .summary)
      LeftBound12060.bound (LeftBound12060.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26010⟩⟩) (rawTerms := some (Proof.Events047.exact12065RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12060.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12077.bound, LeftBound12060.bound]
def bound : CoeffClass := .finite ⟨352054612209664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12077.bound, LeftBound12060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12077.actual selector witness, LeftBound12060.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12257

namespace LeftBound12261
def owner : Owner := ⟨.program ⟨214⟩, ⟨27703⟩⟩
def transferEvent : Nat := 12261
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 12259 .coefficient) (.predecessor 1 12260 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12259 .coefficient)
      LeftBound12254.bound (LeftBound12254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12254.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12260 .coefficient)
      LeftAuthority11961.bound (LeftAuthority11961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11961.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11961.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12254.bound LeftAuthority11961.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12254.bound, LeftAuthority11961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12254.actual selector witness) * (LeftAuthority11961.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12261

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
