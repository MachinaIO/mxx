import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard081

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound13179
def owner : Owner := ⟨.program ⟨214⟩, ⟨13679⟩⟩
def transferEvent : Nat := 13179
def frameStart : Nat := 13129
def rule : BoundRule := .sum [.predecessor 0 13177 .coefficient, .predecessor 1 13178 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13177 .coefficient)
      LeftBound13162.bound (LeftBound13162.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound13162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13178 .coefficient)
      LeftAuthority13175.bound (LeftAuthority13175.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority13175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13162.bound, LeftAuthority13175.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13162.bound, LeftAuthority13175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13162.actual selector witness, LeftAuthority13175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13179

namespace LeftBound13182
def owner : Owner := ⟨.program ⟨214⟩, ⟨13680⟩⟩
def transferEvent : Nat := 13182
def frameStart : Nat := 13129
def rule : BoundRule := .identity (.predecessor 0 13181 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13181 .coefficient)
      LeftBound13179.bound (LeftBound13179.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound13179.derived selector witness)

def rawBound : CoeffClass := LeftBound13179.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound13179.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13182

namespace LeftBound13188
def owner : Owner := ⟨.program ⟨214⟩, ⟨13681⟩⟩
def transferEvent : Nat := 13188
def frameStart : Nat := 13129
def rule : BoundRule := .product (.predecessor 0 13186 .coefficient) (.predecessor 1 13187 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13186 .coefficient)
      LeftAuthority13184.bound (LeftAuthority13184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13184.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13187 .coefficient)
      LeftBound13182.bound (LeftBound13182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority13184.bound LeftBound13182.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13184.bound, LeftBound13182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority13184.actual selector witness) * (LeftBound13182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13188

namespace LeftBound13204
def owner : Owner := ⟨.program ⟨214⟩, ⟨7844⟩⟩
def transferEvent : Nat := 13204
def frameStart : Nat := 13129
def rule : BoundRule := .scale (.predecessor 0 13202 .coefficient) (.value (.predecessor 1 13203 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13202 .coefficient)
      LeftAuthority13200.bound (LeftAuthority13200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13203 .coefficient)
      LeftAuthority13191.bound (LeftAuthority13191.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority13191.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority13200.bound LeftAuthority13191.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13200.bound, LeftAuthority13191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13200.actual selector witness) * (LeftAuthority13191.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound13204

namespace LeftBound13207
def owner : Owner := ⟨.program ⟨214⟩, ⟨6793⟩⟩
def transferEvent : Nat := 13207
def frameStart : Nat := 13129
def rule : BoundRule := .identity (.predecessor 0 13206 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13206 .coefficient)
      LeftAuthority13194.bound (LeftAuthority13194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13194.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13194.derived selector witness)

def rawBound : CoeffClass := LeftAuthority13194.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority13194.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13207

namespace LeftBound13211
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def transferEvent : Nat := 13211
def frameStart : Nat := 13129
def rule : BoundRule := .product (.predecessor 0 13209 .coefficient) (.predecessor 1 13210 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13209 .coefficient)
      LeftBound13207.bound (LeftBound13207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13210 .coefficient)
      LeftBound13204.bound (LeftBound13204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13204.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13207.bound LeftBound13204.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13207.bound, LeftBound13204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13207.actual selector witness) * (LeftBound13204.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13211

namespace LeftBound13216
def owner : Owner := ⟨.program ⟨214⟩, ⟨13682⟩⟩
def transferEvent : Nat := 13216
def frameStart : Nat := 13129
def rule : BoundRule := .sum [.predecessor 0 13214 .coefficient, .predecessor 1 13215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13214 .coefficient)
      LeftBound13211.bound (LeftBound13211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13215 .coefficient)
      LeftBound13188.bound (LeftBound13188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13188.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13211.bound, LeftBound13188.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13211.bound, LeftBound13188.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13211.actual selector witness, LeftBound13188.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13216

namespace LeftBound13220
def owner : Owner := ⟨.program ⟨214⟩, ⟨25858⟩⟩
def transferEvent : Nat := 13220
def frameStart : Nat := 13129
def rule : BoundRule := .product (.predecessor 0 13218 .coefficient) (.predecessor 1 13219 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13218 .coefficient)
      LeftBound13216.bound (LeftBound13216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13219 .coefficient)
      LeftAuthority13173.bound (LeftAuthority13173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13173.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13216.bound LeftAuthority13173.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13216.bound, LeftAuthority13173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13216.actual selector witness) * (LeftAuthority13173.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13220

namespace LeftBound13231
def owner : Owner := ⟨.program ⟨214⟩, ⟨15601⟩⟩
def transferEvent : Nat := 13231
def frameStart : Nat := 13129
def rule : BoundRule := .product (.predecessor 0 13229 .coefficient) (.predecessor 1 13230 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13229 .coefficient)
      LeftAuthority13184.bound (LeftAuthority13184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13184.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13230 .coefficient)
      LeftAuthority13227.bound (LeftAuthority13227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13227.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13227.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13184.bound LeftAuthority13227.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13184.bound, LeftAuthority13227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority13184.actual selector witness) * (LeftAuthority13227.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13231

namespace LeftBound13239
def owner : Owner := ⟨.program ⟨214⟩, ⟨15602⟩⟩
def transferEvent : Nat := 13239
def frameStart : Nat := 13129
def rule : BoundRule := .sum [.predecessor 0 13237 .coefficient, .predecessor 1 13238 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13237 .coefficient)
      LeftAuthority13235.bound (LeftAuthority13235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13238 .coefficient)
      LeftBound13231.bound (LeftBound13231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13231.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority13235.bound, LeftBound13231.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13235.bound, LeftBound13231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority13235.actual selector witness, LeftBound13231.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13239

namespace LeftBound13243
def owner : Owner := ⟨.program ⟨214⟩, ⟨25859⟩⟩
def transferEvent : Nat := 13243
def frameStart : Nat := 13129
def rule : BoundRule := .sum [.predecessor 0 13241 .coefficient, .predecessor 1 13242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13241 .coefficient)
      LeftBound13239.bound (LeftBound13239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13242 .coefficient)
      LeftBound13220.bound (LeftBound13220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13239.bound, LeftBound13220.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13239.bound, LeftBound13220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13239.actual selector witness, LeftBound13220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13243

namespace LeftBound13256
def owner : Owner := ⟨.program ⟨214⟩, ⟨25857⟩⟩
def transferEvent : Nat := 13256
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13254 .coefficient, .predecessor 1 13255 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13254 .coefficient)
      LeftBound13077.bound (LeftBound13077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13077.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13255 .coefficient)
      LeftBound13060.bound (LeftBound13060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13060.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13060.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13077.bound, LeftBound13060.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13077.bound, LeftBound13060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13077.actual selector witness, LeftBound13060.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13256

namespace LeftBound13259
def owner : Owner := ⟨.program ⟨214⟩, ⟨25857⟩⟩
def transferEvent : Nat := 13259
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 13253 .summary, .result 13067 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13253 .summary)
      LeftBound13079.bound (LeftBound13079.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19331⟩⟩) (rawTerms := some (Proof.Events051.exact13253RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13067 .summary)
      LeftBound13062.bound (LeftBound13062.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25856⟩⟩) (rawTerms := some (Proof.Events051.exact13067RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13079.bound, LeftBound13062.bound]
def bound : CoeffClass := .finite ⟨352036291489792, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13079.bound, LeftBound13062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13079.actual selector witness, LeftBound13062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13259

namespace LeftBound13263
def owner : Owner := ⟨.program ⟨214⟩, ⟨27269⟩⟩
def transferEvent : Nat := 13263
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13261 .coefficient) (.predecessor 1 13262 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13261 .coefficient)
      LeftBound13256.bound (LeftBound13256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13262 .coefficient)
      LeftAuthority12963.bound (LeftAuthority12963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12963.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13256.bound LeftAuthority12963.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13256.bound, LeftAuthority12963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13256.actual selector witness) * (LeftAuthority12963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13263

namespace LeftBound13264
def owner : Owner := ⟨.program ⟨214⟩, ⟨27269⟩⟩
def transferEvent : Nat := 13264
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27267⟩⟩]⟩ [⟨.result 12964 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12964 .coefficient)
      LeftAuthority12963.bound (LeftAuthority12963.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27267⟩⟩) (rawTerms := some (Proof.Events050.exact12964RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12963.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12963.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12963.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13264

namespace LeftBound13265
def owner : Owner := ⟨.program ⟨214⟩, ⟨27269⟩⟩
def transferEvent : Nat := 13265
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 13260 .summary) (.transfer 13264) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13260 .summary)
      LeftBound13259.bound (LeftBound13259.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25857⟩⟩) (rawTerms := some (Proof.Events051.exact13260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13264)
      LeftBound13264.bound (LeftBound13264.actual selector witness) := by
  exact .transfer (LeftBound13264.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13259.bound LeftBound13264.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13259.bound, LeftBound13264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13259.actual selector witness) * (LeftBound13264.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13265

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
