import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard089

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound14190
def owner : Owner := ⟨.program ⟨214⟩, ⟨11091⟩⟩
def transferEvent : Nat := 14190
def frameStart : Nat := 14131
def rule : BoundRule := .product (.predecessor 0 14188 .coefficient) (.predecessor 1 14189 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14188 .coefficient)
      LeftAuthority14186.bound (LeftAuthority14186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14189 .coefficient)
      LeftBound14184.bound (LeftBound14184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14184.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority14186.bound LeftBound14184.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14186.bound, LeftBound14184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority14186.actual selector witness) * (LeftBound14184.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14190

namespace LeftBound14206
def owner : Owner := ⟨.program ⟨214⟩, ⟨7838⟩⟩
def transferEvent : Nat := 14206
def frameStart : Nat := 14131
def rule : BoundRule := .scale (.predecessor 0 14204 .coefficient) (.value (.predecessor 1 14205 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14204 .coefficient)
      LeftAuthority14202.bound (LeftAuthority14202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14205 .coefficient)
      LeftAuthority14193.bound (LeftAuthority14193.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority14193.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority14202.bound LeftAuthority14193.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14202.bound, LeftAuthority14193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14202.actual selector witness) * (LeftAuthority14193.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound14206

namespace LeftBound14209
def owner : Owner := ⟨.program ⟨214⟩, ⟨6791⟩⟩
def transferEvent : Nat := 14209
def frameStart : Nat := 14131
def rule : BoundRule := .identity (.predecessor 0 14208 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14208 .coefficient)
      LeftAuthority14196.bound (LeftAuthority14196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14196.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14196.derived selector witness)

def rawBound : CoeffClass := LeftAuthority14196.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority14196.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound14209

namespace LeftBound14213
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def transferEvent : Nat := 14213
def frameStart : Nat := 14131
def rule : BoundRule := .product (.predecessor 0 14211 .coefficient) (.predecessor 1 14212 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14211 .coefficient)
      LeftBound14209.bound (LeftBound14209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14212 .coefficient)
      LeftBound14206.bound (LeftBound14206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14206.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14209.bound LeftBound14206.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14209.bound, LeftBound14206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14209.actual selector witness) * (LeftBound14206.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14213

namespace LeftBound14218
def owner : Owner := ⟨.program ⟨214⟩, ⟨11092⟩⟩
def transferEvent : Nat := 14218
def frameStart : Nat := 14131
def rule : BoundRule := .sum [.predecessor 0 14216 .coefficient, .predecessor 1 14217 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14216 .coefficient)
      LeftBound14213.bound (LeftBound14213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14217 .coefficient)
      LeftBound14190.bound (LeftBound14190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14190.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14190.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14213.bound, LeftBound14190.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14213.bound, LeftBound14190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14213.actual selector witness, LeftBound14190.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14218

namespace LeftBound14222
def owner : Owner := ⟨.program ⟨214⟩, ⟨25088⟩⟩
def transferEvent : Nat := 14222
def frameStart : Nat := 14131
def rule : BoundRule := .product (.predecessor 0 14220 .coefficient) (.predecessor 1 14221 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14220 .coefficient)
      LeftBound14218.bound (LeftBound14218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14221 .coefficient)
      LeftAuthority14175.bound (LeftAuthority14175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14175.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14218.bound LeftAuthority14175.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14218.bound, LeftAuthority14175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14218.actual selector witness) * (LeftAuthority14175.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14222

namespace LeftBound14233
def owner : Owner := ⟨.program ⟨214⟩, ⟨15132⟩⟩
def transferEvent : Nat := 14233
def frameStart : Nat := 14131
def rule : BoundRule := .product (.predecessor 0 14231 .coefficient) (.predecessor 1 14232 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14231 .coefficient)
      LeftAuthority14186.bound (LeftAuthority14186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14232 .coefficient)
      LeftAuthority14229.bound (LeftAuthority14229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14229.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14229.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14186.bound LeftAuthority14229.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14186.bound, LeftAuthority14229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority14186.actual selector witness) * (LeftAuthority14229.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14233

namespace LeftBound14241
def owner : Owner := ⟨.program ⟨214⟩, ⟨15133⟩⟩
def transferEvent : Nat := 14241
def frameStart : Nat := 14131
def rule : BoundRule := .sum [.predecessor 0 14239 .coefficient, .predecessor 1 14240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14239 .coefficient)
      LeftAuthority14237.bound (LeftAuthority14237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14237.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14240 .coefficient)
      LeftBound14233.bound (LeftBound14233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14233.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority14237.bound, LeftBound14233.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14237.bound, LeftBound14233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority14237.actual selector witness, LeftBound14233.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14241

namespace LeftBound14245
def owner : Owner := ⟨.program ⟨214⟩, ⟨25089⟩⟩
def transferEvent : Nat := 14245
def frameStart : Nat := 14131
def rule : BoundRule := .sum [.predecessor 0 14243 .coefficient, .predecessor 1 14244 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14243 .coefficient)
      LeftBound14241.bound (LeftBound14241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14244 .coefficient)
      LeftBound14222.bound (LeftBound14222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14222.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14222.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14241.bound, LeftBound14222.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14241.bound, LeftBound14222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14241.actual selector witness, LeftBound14222.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14245

namespace LeftBound14258
def owner : Owner := ⟨.program ⟨214⟩, ⟨25087⟩⟩
def transferEvent : Nat := 14258
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14256 .coefficient, .predecessor 1 14257 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14256 .coefficient)
      LeftBound14079.bound (LeftBound14079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14257 .coefficient)
      LeftBound14062.bound (LeftBound14062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14079.bound, LeftBound14062.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14079.bound, LeftBound14062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14079.actual selector witness, LeftBound14062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14258

namespace LeftBound14261
def owner : Owner := ⟨.program ⟨214⟩, ⟨25087⟩⟩
def transferEvent : Nat := 14261
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 14255 .summary, .result 14069 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14255 .summary)
      LeftBound14081.bound (LeftBound14081.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19187⟩⟩) (rawTerms := some (Proof.Events055.exact14255RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14069 .summary)
      LeftBound14064.bound (LeftBound14064.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25086⟩⟩) (rawTerms := some (Proof.Events054.exact14069RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14064.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14081.bound, LeftBound14064.bound]
def bound : CoeffClass := .finite ⟨352017970769920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14081.bound, LeftBound14064.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14081.actual selector witness, LeftBound14064.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14261

namespace LeftBound14265
def owner : Owner := ⟨.program ⟨214⟩, ⟨26835⟩⟩
def transferEvent : Nat := 14265
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14263 .coefficient) (.predecessor 1 14264 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14263 .coefficient)
      LeftBound14258.bound (LeftBound14258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14264 .coefficient)
      LeftAuthority13965.bound (LeftAuthority13965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13965.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13965.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14258.bound LeftAuthority13965.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14258.bound, LeftAuthority13965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14258.actual selector witness) * (LeftAuthority13965.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14265

namespace LeftBound14266
def owner : Owner := ⟨.program ⟨214⟩, ⟨26835⟩⟩
def transferEvent : Nat := 14266
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26833⟩⟩]⟩ [⟨.result 13966 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13966 .coefficient)
      LeftAuthority13965.bound (LeftAuthority13965.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26833⟩⟩) (rawTerms := some (Proof.Events054.exact13966RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13965.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13965.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13965.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13965.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound14266

namespace LeftBound14267
def owner : Owner := ⟨.program ⟨214⟩, ⟨26835⟩⟩
def transferEvent : Nat := 14267
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 14262 .summary) (.transfer 14266) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14262 .summary)
      LeftBound14261.bound (LeftBound14261.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25087⟩⟩) (rawTerms := some (Proof.Events055.exact14262RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 14266)
      LeftBound14266.bound (LeftBound14266.actual selector witness) := by
  exact .transfer (LeftBound14266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14261.bound LeftBound14266.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14261.bound, LeftBound14266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14261.actual selector witness) * (LeftBound14266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14267

namespace LeftBound14278
def owner : Owner := ⟨.program ⟨214⟩, ⟨20698⟩⟩
def transferEvent : Nat := 14278
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 14276 .coefficient) (.value (.predecessor 1 14277 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14276 .coefficient)
      LeftAuthority14274.bound (LeftAuthority14274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14274.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14277 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority14274.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14274.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14274.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound14278

namespace LeftBound14282
def owner : Owner := ⟨.program ⟨214⟩, ⟨20699⟩⟩
def transferEvent : Nat := 14282
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14280 .coefficient) (.predecessor 1 14281 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14280 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14281 .coefficient)
      LeftBound14278.bound (LeftBound14278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14278.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound14278.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound14278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound14278.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14282

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
