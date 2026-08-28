import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound52207
def owner : Owner := ⟨.program ⟨214⟩, ⟨20038⟩⟩
def transferEvent : Nat := 52207
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 52205 .coefficient) (.value (.predecessor 1 52206 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52205 .coefficient)
      LeftAuthority52203.bound (LeftAuthority52203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52206 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority52203.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52203.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52203.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound52207

namespace LeftBound52211
def owner : Owner := ⟨.program ⟨214⟩, ⟨20039⟩⟩
def transferEvent : Nat := 52211
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52209 .coefficient) (.predecessor 1 52210 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52209 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52210 .coefficient)
      LeftBound52207.bound (LeftBound52207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52207.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound52207.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound52207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound52207.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52211

namespace LeftBound52212
def owner : Owner := ⟨.program ⟨214⟩, ⟨20039⟩⟩
def transferEvent : Nat := 52212
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20036⟩⟩]⟩ [⟨.result 52204 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52204 .coefficient)
      LeftAuthority52203.bound (LeftAuthority52203.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20036⟩⟩) (rawTerms := some (Proof.Events203.exact52204RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52203.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority52203.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52203.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound52212

namespace LeftBound52213
def owner : Owner := ⟨.program ⟨214⟩, ⟨20039⟩⟩
def transferEvent : Nat := 52213
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 52212) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 52212)
      LeftBound52212.bound (LeftBound52212.actual selector witness) := by
  exact .transfer (LeftBound52212.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound52212.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound52212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound52212.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52213

namespace LeftBound52292
def owner : Owner := ⟨.program ⟨214⟩, ⟨12771⟩⟩
def transferEvent : Nat := 52292
def frameStart : Nat := 52263
def rule : BoundRule := .product (.predecessor 0 52290 .coefficient) (.predecessor 1 52291 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52290 .coefficient)
      LeftAuthority52288.bound (LeftAuthority52288.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52288.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52288.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52291 .coefficient)
      LeftAuthority52285.bound (LeftAuthority52285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52285.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52285.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority52288.bound LeftAuthority52285.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52288.bound, LeftAuthority52285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority52288.actual selector witness) * (LeftAuthority52285.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52292

namespace LeftBound52296
def owner : Owner := ⟨.program ⟨214⟩, ⟨12772⟩⟩
def transferEvent : Nat := 52296
def frameStart : Nat := 52263
def rule : BoundRule := .identity (.predecessor 0 52295 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52295 .coefficient)
      LeftBound52292.bound (LeftBound52292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52292.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52292.derived selector witness)

def rawBound : CoeffClass := LeftBound52292.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound52292.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52296

namespace LeftBound52313
def owner : Owner := ⟨.program ⟨214⟩, ⟨12862⟩⟩
def transferEvent : Nat := 52313
def frameStart : Nat := 52263
def rule : BoundRule := .sum [.predecessor 0 52311 .coefficient, .predecessor 1 52312 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52311 .coefficient)
      LeftBound52296.bound (LeftBound52296.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52296.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52312 .coefficient)
      LeftAuthority52309.bound (LeftAuthority52309.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority52309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52296.bound, LeftAuthority52309.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52296.bound, LeftAuthority52309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52296.actual selector witness, LeftAuthority52309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52313

namespace LeftBound52316
def owner : Owner := ⟨.program ⟨214⟩, ⟨12863⟩⟩
def transferEvent : Nat := 52316
def frameStart : Nat := 52263
def rule : BoundRule := .identity (.predecessor 0 52315 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52315 .coefficient)
      LeftBound52313.bound (LeftBound52313.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52313.derived selector witness)

def rawBound : CoeffClass := LeftBound52313.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound52313.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52316

namespace LeftBound52322
def owner : Owner := ⟨.program ⟨214⟩, ⟨12864⟩⟩
def transferEvent : Nat := 52322
def frameStart : Nat := 52263
def rule : BoundRule := .product (.predecessor 0 52320 .coefficient) (.predecessor 1 52321 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52320 .coefficient)
      LeftAuthority52318.bound (LeftAuthority52318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52321 .coefficient)
      LeftBound52316.bound (LeftBound52316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52316.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority52318.bound LeftBound52316.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52318.bound, LeftBound52316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority52318.actual selector witness) * (LeftBound52316.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52322

namespace LeftBound52338
def owner : Owner := ⟨.program ⟨214⟩, ⟨7874⟩⟩
def transferEvent : Nat := 52338
def frameStart : Nat := 52263
def rule : BoundRule := .scale (.predecessor 0 52336 .coefficient) (.value (.predecessor 1 52337 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52336 .coefficient)
      LeftAuthority52334.bound (LeftAuthority52334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52337 .coefficient)
      LeftAuthority52325.bound (LeftAuthority52325.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority52325.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority52334.bound LeftAuthority52325.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52334.bound, LeftAuthority52325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52334.actual selector witness) * (LeftAuthority52325.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound52338

namespace LeftBound52341
def owner : Owner := ⟨.program ⟨214⟩, ⟨6767⟩⟩
def transferEvent : Nat := 52341
def frameStart : Nat := 52263
def rule : BoundRule := .identity (.predecessor 0 52340 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52340 .coefficient)
      LeftAuthority52328.bound (LeftAuthority52328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52328.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52328.derived selector witness)

def rawBound : CoeffClass := LeftAuthority52328.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority52328.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52341

namespace LeftBound52345
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def transferEvent : Nat := 52345
def frameStart : Nat := 52263
def rule : BoundRule := .product (.predecessor 0 52343 .coefficient) (.predecessor 1 52344 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52343 .coefficient)
      LeftBound52341.bound (LeftBound52341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52344 .coefficient)
      LeftBound52338.bound (LeftBound52338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52338.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52341.bound LeftBound52338.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52341.bound, LeftBound52338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52341.actual selector witness) * (LeftBound52338.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52345

namespace LeftBound52350
def owner : Owner := ⟨.program ⟨214⟩, ⟨12865⟩⟩
def transferEvent : Nat := 52350
def frameStart : Nat := 52263
def rule : BoundRule := .sum [.predecessor 0 52348 .coefficient, .predecessor 1 52349 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52348 .coefficient)
      LeftBound52345.bound (LeftBound52345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52349 .coefficient)
      LeftBound52322.bound (LeftBound52322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52345.bound, LeftBound52322.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52345.bound, LeftBound52322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52345.actual selector witness, LeftBound52322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52350

namespace LeftBound52354
def owner : Owner := ⟨.program ⟨214⟩, ⟨25535⟩⟩
def transferEvent : Nat := 52354
def frameStart : Nat := 52263
def rule : BoundRule := .product (.predecessor 0 52352 .coefficient) (.predecessor 1 52353 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52352 .coefficient)
      LeftBound52350.bound (LeftBound52350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52350.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52353 .coefficient)
      LeftAuthority52307.bound (LeftAuthority52307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52307.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52350.bound LeftAuthority52307.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52350.bound, LeftAuthority52307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52350.actual selector witness) * (LeftAuthority52307.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52354

namespace LeftBound52365
def owner : Owner := ⟨.program ⟨214⟩, ⟨16639⟩⟩
def transferEvent : Nat := 52365
def frameStart : Nat := 52263
def rule : BoundRule := .product (.predecessor 0 52363 .coefficient) (.predecessor 1 52364 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52363 .coefficient)
      LeftAuthority52318.bound (LeftAuthority52318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52364 .coefficient)
      LeftAuthority52361.bound (LeftAuthority52361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52361.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52361.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority52318.bound LeftAuthority52361.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52318.bound, LeftAuthority52361.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority52318.actual selector witness) * (LeftAuthority52361.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52365

namespace LeftBound52373
def owner : Owner := ⟨.program ⟨214⟩, ⟨16640⟩⟩
def transferEvent : Nat := 52373
def frameStart : Nat := 52263
def rule : BoundRule := .sum [.predecessor 0 52371 .coefficient, .predecessor 1 52372 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52371 .coefficient)
      LeftAuthority52369.bound (LeftAuthority52369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52369.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52372 .coefficient)
      LeftBound52365.bound (LeftBound52365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52365.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52365.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority52369.bound, LeftBound52365.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52369.bound, LeftBound52365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority52369.actual selector witness, LeftBound52365.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52373

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
