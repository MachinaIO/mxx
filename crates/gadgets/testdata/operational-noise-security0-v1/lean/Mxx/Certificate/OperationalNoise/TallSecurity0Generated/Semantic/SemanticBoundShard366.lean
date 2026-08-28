import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54250
def owner : Owner := ⟨.program ⟨214⟩, ⟨11863⟩⟩
def transferEvent : Nat := 54250
def frameStart : Nat := 54191
def rule : BoundRule := .product (.predecessor 0 54248 .coefficient) (.predecessor 1 54249 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54248 .coefficient)
      LeftAuthority54246.bound (LeftAuthority54246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54249 .coefficient)
      LeftBound54244.bound (LeftBound54244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority54246.bound LeftBound54244.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54246.bound, LeftBound54244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority54246.actual selector witness) * (LeftBound54244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54250

namespace LeftBound54266
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def transferEvent : Nat := 54266
def frameStart : Nat := 54191
def rule : BoundRule := .scale (.predecessor 0 54264 .coefficient) (.value (.predecessor 1 54265 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54264 .coefficient)
      LeftAuthority54262.bound (LeftAuthority54262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54265 .coefficient)
      LeftAuthority54253.bound (LeftAuthority54253.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority54253.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority54262.bound LeftAuthority54253.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54262.bound, LeftAuthority54253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54262.actual selector witness) * (LeftAuthority54253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54266

namespace LeftBound54269
def owner : Owner := ⟨.program ⟨214⟩, ⟨6763⟩⟩
def transferEvent : Nat := 54269
def frameStart : Nat := 54191
def rule : BoundRule := .identity (.predecessor 0 54268 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54268 .coefficient)
      LeftAuthority54256.bound (LeftAuthority54256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54256.derived selector witness)

def rawBound : CoeffClass := LeftAuthority54256.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority54256.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54269

namespace LeftBound54273
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def transferEvent : Nat := 54273
def frameStart : Nat := 54191
def rule : BoundRule := .product (.predecessor 0 54271 .coefficient) (.predecessor 1 54272 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54271 .coefficient)
      LeftBound54269.bound (LeftBound54269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54269.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54272 .coefficient)
      LeftBound54266.bound (LeftBound54266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54269.bound LeftBound54266.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54269.bound, LeftBound54266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54269.actual selector witness) * (LeftBound54266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54273

namespace LeftBound54278
def owner : Owner := ⟨.program ⟨214⟩, ⟨11864⟩⟩
def transferEvent : Nat := 54278
def frameStart : Nat := 54191
def rule : BoundRule := .sum [.predecessor 0 54276 .coefficient, .predecessor 1 54277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54276 .coefficient)
      LeftBound54273.bound (LeftBound54273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54277 .coefficient)
      LeftBound54250.bound (LeftBound54250.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54250.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54250.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54273.bound, LeftBound54250.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54273.bound, LeftBound54250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54273.actual selector witness, LeftBound54250.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54278

namespace LeftBound54282
def owner : Owner := ⟨.program ⟨214⟩, ⟨25150⟩⟩
def transferEvent : Nat := 54282
def frameStart : Nat := 54191
def rule : BoundRule := .product (.predecessor 0 54280 .coefficient) (.predecessor 1 54281 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54280 .coefficient)
      LeftBound54278.bound (LeftBound54278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54281 .coefficient)
      LeftAuthority54235.bound (LeftAuthority54235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54278.bound LeftAuthority54235.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54278.bound, LeftAuthority54235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54278.actual selector witness) * (LeftAuthority54235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54282

namespace LeftBound54293
def owner : Owner := ⟨.program ⟨214⟩, ⟨16268⟩⟩
def transferEvent : Nat := 54293
def frameStart : Nat := 54191
def rule : BoundRule := .product (.predecessor 0 54291 .coefficient) (.predecessor 1 54292 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54291 .coefficient)
      LeftAuthority54246.bound (LeftAuthority54246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54292 .coefficient)
      LeftAuthority54289.bound (LeftAuthority54289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority54246.bound LeftAuthority54289.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54246.bound, LeftAuthority54289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority54246.actual selector witness) * (LeftAuthority54289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54293

namespace LeftBound54301
def owner : Owner := ⟨.program ⟨214⟩, ⟨16269⟩⟩
def transferEvent : Nat := 54301
def frameStart : Nat := 54191
def rule : BoundRule := .sum [.predecessor 0 54299 .coefficient, .predecessor 1 54300 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54299 .coefficient)
      LeftAuthority54297.bound (LeftAuthority54297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54297.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54300 .coefficient)
      LeftBound54293.bound (LeftBound54293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54293.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority54297.bound, LeftBound54293.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54297.bound, LeftBound54293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority54297.actual selector witness, LeftBound54293.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54301

namespace LeftBound54305
def owner : Owner := ⟨.program ⟨214⟩, ⟨25151⟩⟩
def transferEvent : Nat := 54305
def frameStart : Nat := 54191
def rule : BoundRule := .sum [.predecessor 0 54303 .coefficient, .predecessor 1 54304 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54303 .coefficient)
      LeftBound54301.bound (LeftBound54301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54304 .coefficient)
      LeftBound54282.bound (LeftBound54282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54301.bound, LeftBound54282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54301.bound, LeftBound54282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54301.actual selector witness, LeftBound54282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54305

namespace LeftBound54318
def owner : Owner := ⟨.program ⟨214⟩, ⟨25149⟩⟩
def transferEvent : Nat := 54318
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54316 .coefficient, .predecessor 1 54317 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54316 .coefficient)
      LeftBound54139.bound (LeftBound54139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54317 .coefficient)
      LeftBound54122.bound (LeftBound54122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54122.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54139.bound, LeftBound54122.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54139.bound, LeftBound54122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54139.actual selector witness, LeftBound54122.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54318

namespace LeftBound54321
def owner : Owner := ⟨.program ⟨214⟩, ⟨25149⟩⟩
def transferEvent : Nat := 54321
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 54315 .summary, .result 54129 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54315 .summary)
      LeftBound54141.bound (LeftBound54141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19751⟩⟩) (rawTerms := some (Proof.Events212.exact54315RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54129 .summary)
      LeftBound54124.bound (LeftBound54124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25148⟩⟩) (rawTerms := some (Proof.Events211.exact54129RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54124.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54141.bound, LeftBound54124.bound]
def bound : CoeffClass := .finite ⟨352097360556032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54141.bound, LeftBound54124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54141.actual selector witness, LeftBound54124.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54321

namespace LeftBound54325
def owner : Owner := ⟨.program ⟨214⟩, ⟨28532⟩⟩
def transferEvent : Nat := 54325
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54323 .coefficient) (.predecessor 1 54324 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54323 .coefficient)
      LeftBound54318.bound (LeftBound54318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54324 .coefficient)
      LeftAuthority54044.bound (LeftAuthority54044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54044.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54044.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54318.bound LeftAuthority54044.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54318.bound, LeftAuthority54044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54318.actual selector witness) * (LeftAuthority54044.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54325

namespace LeftBound54326
def owner : Owner := ⟨.program ⟨214⟩, ⟨28532⟩⟩
def transferEvent : Nat := 54326
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28530⟩⟩]⟩ [⟨.result 54045 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54045 .coefficient)
      LeftAuthority54044.bound (LeftAuthority54044.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28530⟩⟩) (rawTerms := some (Proof.Events211.exact54045RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54044.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54044.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority54044.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54044.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54326

namespace LeftBound54327
def owner : Owner := ⟨.program ⟨214⟩, ⟨28532⟩⟩
def transferEvent : Nat := 54327
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54322 .summary) (.transfer 54326) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54322 .summary)
      LeftBound54321.bound (LeftBound54321.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25149⟩⟩) (rawTerms := some (Proof.Events212.exact54322RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54326)
      LeftBound54326.bound (LeftBound54326.actual selector witness) := by
  exact .transfer (LeftBound54326.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54321.bound LeftBound54326.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54321.bound, LeftBound54326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54321.actual selector witness) * (LeftBound54326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54327

namespace LeftBound54338
def owner : Owner := ⟨.program ⟨214⟩, ⟨21838⟩⟩
def transferEvent : Nat := 54338
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 54336 .coefficient) (.value (.predecessor 1 54337 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54336 .coefficient)
      LeftAuthority54334.bound (LeftAuthority54334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54337 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority54334.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54334.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54334.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54338

namespace LeftBound54342
def owner : Owner := ⟨.program ⟨214⟩, ⟨21839⟩⟩
def transferEvent : Nat := 54342
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54340 .coefficient) (.predecessor 1 54341 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54340 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54341 .coefficient)
      LeftBound54338.bound (LeftBound54338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54338.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound54338.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound54338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound54338.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54342

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
