import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard499

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound73207
def owner : Owner := ⟨.program ⟨214⟩, ⟨10769⟩⟩
def transferEvent : Nat := 73207
def frameStart : Nat := 73154
def rule : BoundRule := .identity (.predecessor 0 73206 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73206 .coefficient)
      LeftBound73204.bound (LeftBound73204.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound73204.derived selector witness)

def rawBound : CoeffClass := LeftBound73204.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound73204.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound73207

namespace LeftBound73213
def owner : Owner := ⟨.program ⟨214⟩, ⟨10770⟩⟩
def transferEvent : Nat := 73213
def frameStart : Nat := 73154
def rule : BoundRule := .product (.predecessor 0 73211 .coefficient) (.predecessor 1 73212 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73211 .coefficient)
      LeftAuthority73209.bound (LeftAuthority73209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73209.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73212 .coefficient)
      LeftBound73207.bound (LeftBound73207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73207.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority73209.bound LeftBound73207.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73209.bound, LeftBound73207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority73209.actual selector witness) * (LeftBound73207.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73213

namespace LeftBound73229
def owner : Owner := ⟨.program ⟨214⟩, ⟨7835⟩⟩
def transferEvent : Nat := 73229
def frameStart : Nat := 73154
def rule : BoundRule := .scale (.predecessor 0 73227 .coefficient) (.value (.predecessor 1 73228 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73227 .coefficient)
      LeftAuthority73225.bound (LeftAuthority73225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73225.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73228 .coefficient)
      LeftAuthority73216.bound (LeftAuthority73216.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority73216.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority73225.bound LeftAuthority73216.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73225.bound, LeftAuthority73216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73225.actual selector witness) * (LeftAuthority73216.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound73229

namespace LeftBound73232
def owner : Owner := ⟨.program ⟨214⟩, ⟨6782⟩⟩
def transferEvent : Nat := 73232
def frameStart : Nat := 73154
def rule : BoundRule := .identity (.predecessor 0 73231 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73231 .coefficient)
      LeftAuthority73219.bound (LeftAuthority73219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73219.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73219.derived selector witness)

def rawBound : CoeffClass := LeftAuthority73219.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority73219.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound73232

namespace LeftBound73236
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def transferEvent : Nat := 73236
def frameStart : Nat := 73154
def rule : BoundRule := .product (.predecessor 0 73234 .coefficient) (.predecessor 1 73235 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73234 .coefficient)
      LeftBound73232.bound (LeftBound73232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73232.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73235 .coefficient)
      LeftBound73229.bound (LeftBound73229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73229.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73232.bound LeftBound73229.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73232.bound, LeftBound73229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73232.actual selector witness) * (LeftBound73229.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73236

namespace LeftBound73241
def owner : Owner := ⟨.program ⟨214⟩, ⟨10771⟩⟩
def transferEvent : Nat := 73241
def frameStart : Nat := 73154
def rule : BoundRule := .sum [.predecessor 0 73239 .coefficient, .predecessor 1 73240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73239 .coefficient)
      LeftBound73236.bound (LeftBound73236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73240 .coefficient)
      LeftBound73213.bound (LeftBound73213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73213.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73236.bound, LeftBound73213.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73236.bound, LeftBound73213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73236.actual selector witness, LeftBound73213.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73241

namespace LeftBound73245
def owner : Owner := ⟨.program ⟨214⟩, ⟨24986⟩⟩
def transferEvent : Nat := 73245
def frameStart : Nat := 73154
def rule : BoundRule := .product (.predecessor 0 73243 .coefficient) (.predecessor 1 73244 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73243 .coefficient)
      LeftBound73241.bound (LeftBound73241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73244 .coefficient)
      LeftAuthority73198.bound (LeftAuthority73198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73198.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73241.bound LeftAuthority73198.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73241.bound, LeftAuthority73198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73241.actual selector witness) * (LeftAuthority73198.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73245

namespace LeftBound73256
def owner : Owner := ⟨.program ⟨214⟩, ⟨14951⟩⟩
def transferEvent : Nat := 73256
def frameStart : Nat := 73154
def rule : BoundRule := .product (.predecessor 0 73254 .coefficient) (.predecessor 1 73255 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73254 .coefficient)
      LeftAuthority73209.bound (LeftAuthority73209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73209.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73255 .coefficient)
      LeftAuthority73252.bound (LeftAuthority73252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73252.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73252.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority73209.bound LeftAuthority73252.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73209.bound, LeftAuthority73252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority73209.actual selector witness) * (LeftAuthority73252.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73256

namespace LeftBound73264
def owner : Owner := ⟨.program ⟨214⟩, ⟨14952⟩⟩
def transferEvent : Nat := 73264
def frameStart : Nat := 73154
def rule : BoundRule := .sum [.predecessor 0 73262 .coefficient, .predecessor 1 73263 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73262 .coefficient)
      LeftAuthority73260.bound (LeftAuthority73260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73260.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73263 .coefficient)
      LeftBound73256.bound (LeftBound73256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73256.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority73260.bound, LeftBound73256.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73260.bound, LeftBound73256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority73260.actual selector witness, LeftBound73256.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73264

namespace LeftBound73268
def owner : Owner := ⟨.program ⟨214⟩, ⟨24987⟩⟩
def transferEvent : Nat := 73268
def frameStart : Nat := 73154
def rule : BoundRule := .sum [.predecessor 0 73266 .coefficient, .predecessor 1 73267 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73266 .coefficient)
      LeftBound73264.bound (LeftBound73264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73267 .coefficient)
      LeftBound73245.bound (LeftBound73245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73245.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73264.bound, LeftBound73245.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73264.bound, LeftBound73245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73264.actual selector witness, LeftBound73245.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73268

namespace LeftBound73281
def owner : Owner := ⟨.program ⟨214⟩, ⟨24985⟩⟩
def transferEvent : Nat := 73281
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73279 .coefficient, .predecessor 1 73280 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73279 .coefficient)
      LeftBound73102.bound (LeftBound73102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73280 .coefficient)
      LeftBound73085.bound (LeftBound73085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73102.bound, LeftBound73085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73102.bound, LeftBound73085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73102.actual selector witness, LeftBound73085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73281

namespace LeftBound73284
def owner : Owner := ⟨.program ⟨214⟩, ⟨24985⟩⟩
def transferEvent : Nat := 73284
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73278 .summary, .result 73092 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73278 .summary)
      LeftBound73104.bound (LeftBound73104.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19095⟩⟩) (rawTerms := some (Proof.Events286.exact73278RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73092 .summary)
      LeftBound73087.bound (LeftBound73087.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24984⟩⟩) (rawTerms := some (Proof.Events285.exact73092RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73104.bound, LeftBound73087.bound]
def bound : CoeffClass := .finite ⟨352014917316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73104.bound, LeftBound73087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73104.actual selector witness, LeftBound73087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73284

namespace LeftBound73288
def owner : Owner := ⟨.program ⟨214⟩, ⟨26553⟩⟩
def transferEvent : Nat := 73288
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73286 .coefficient) (.predecessor 1 73287 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73286 .coefficient)
      LeftBound73281.bound (LeftBound73281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73287 .coefficient)
      LeftAuthority73007.bound (LeftAuthority73007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73007.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73007.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73281.bound LeftAuthority73007.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73281.bound, LeftAuthority73007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73281.actual selector witness) * (LeftAuthority73007.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73288

namespace LeftBound73289
def owner : Owner := ⟨.program ⟨214⟩, ⟨26553⟩⟩
def transferEvent : Nat := 73289
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩ [⟨.result 73008 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73008 .coefficient)
      LeftAuthority73007.bound (LeftAuthority73007.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26551⟩⟩) (rawTerms := some (Proof.Events285.exact73008RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73007.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73007.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority73007.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73007.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73289

namespace LeftBound73290
def owner : Owner := ⟨.program ⟨214⟩, ⟨26553⟩⟩
def transferEvent : Nat := 73290
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 73285 .summary) (.transfer 73289) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73285 .summary)
      LeftBound73284.bound (LeftBound73284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24985⟩⟩) (rawTerms := some (Proof.Events286.exact73285RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 73289)
      LeftBound73289.bound (LeftBound73289.actual selector witness) := by
  exact .transfer (LeftBound73289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73284.bound LeftBound73289.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73284.bound, LeftBound73289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73284.actual selector witness) * (LeftBound73289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73290

namespace LeftBound73301
def owner : Owner := ⟨.program ⟨214⟩, ⟨20534⟩⟩
def transferEvent : Nat := 73301
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 73299 .coefficient) (.value (.predecessor 1 73300 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73299 .coefficient)
      LeftAuthority73297.bound (LeftAuthority73297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73297.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73300 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority73297.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73297.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73297.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound73301

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
