import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard318

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound48066
def owner : Owner := ⟨.program ⟨214⟩, ⟨21627⟩⟩
def transferEvent : Nat := 48066
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩ [⟨.result 48058 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48058 .coefficient)
      LeftAuthority48057.bound (LeftAuthority48057.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21624⟩⟩) (rawTerms := some (Proof.Events187.exact48058RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48057.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48057.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48057.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48066

namespace LeftBound48067
def owner : Owner := ⟨.program ⟨214⟩, ⟨21627⟩⟩
def transferEvent : Nat := 48067
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 48066) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48066)
      LeftBound48066.bound (LeftBound48066.actual selector witness) := by
  exact .transfer (LeftBound48066.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound48066.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound48066.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound48066.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48067

namespace LeftBound48162
def owner : Owner := ⟨.program ⟨214⟩, ⟨16187⟩⟩
def transferEvent : Nat := 48162
def frameStart : Nat := 48123
def rule : BoundRule := .identity (.predecessor 0 48161 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48161 .coefficient)
      LeftAuthority48159.bound (LeftAuthority48159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48159.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48159.derived selector witness)

def rawBound : CoeffClass := LeftAuthority48159.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority48159.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48162

namespace LeftBound48179
def owner : Owner := ⟨.program ⟨214⟩, ⟨16226⟩⟩
def transferEvent : Nat := 48179
def frameStart : Nat := 48123
def rule : BoundRule := .sum [.predecessor 0 48177 .coefficient, .predecessor 1 48178 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48177 .coefficient)
      LeftBound48162.bound (LeftBound48162.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48178 .coefficient)
      LeftAuthority48175.bound (LeftAuthority48175.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority48175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48162.bound, LeftAuthority48175.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48162.bound, LeftAuthority48175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48162.actual selector witness, LeftAuthority48175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48179

namespace LeftBound48182
def owner : Owner := ⟨.program ⟨214⟩, ⟨16227⟩⟩
def transferEvent : Nat := 48182
def frameStart : Nat := 48123
def rule : BoundRule := .identity (.predecessor 0 48181 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48181 .coefficient)
      LeftBound48179.bound (LeftBound48179.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48179.derived selector witness)

def rawBound : CoeffClass := LeftBound48179.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound48179.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48182

namespace LeftBound48188
def owner : Owner := ⟨.program ⟨214⟩, ⟨16228⟩⟩
def transferEvent : Nat := 48188
def frameStart : Nat := 48123
def rule : BoundRule := .product (.predecessor 0 48186 .coefficient) (.predecessor 1 48187 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48186 .coefficient)
      LeftAuthority48184.bound (LeftAuthority48184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48184.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48187 .coefficient)
      LeftBound48182.bound (LeftBound48182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority48184.bound LeftBound48182.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48184.bound, LeftBound48182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority48184.actual selector witness) * (LeftBound48182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48188

namespace LeftBound48196
def owner : Owner := ⟨.program ⟨214⟩, ⟨16229⟩⟩
def transferEvent : Nat := 48196
def frameStart : Nat := 48123
def rule : BoundRule := .sum [.predecessor 0 48194 .coefficient, .predecessor 1 48195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48194 .coefficient)
      LeftAuthority48192.bound (LeftAuthority48192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48195 .coefficient)
      LeftBound48188.bound (LeftBound48188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48188.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48192.bound, LeftBound48188.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48192.bound, LeftBound48188.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority48192.actual selector witness, LeftBound48188.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48196

namespace LeftBound48200
def owner : Owner := ⟨.program ⟨214⟩, ⟨28320⟩⟩
def transferEvent : Nat := 48200
def frameStart : Nat := 48123
def rule : BoundRule := .product (.predecessor 0 48198 .coefficient) (.predecessor 1 48199 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48198 .coefficient)
      LeftBound48196.bound (LeftBound48196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48199 .coefficient)
      LeftAuthority48173.bound (LeftAuthority48173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48173.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48196.bound LeftAuthority48173.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48196.bound, LeftAuthority48173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48196.actual selector witness) * (LeftAuthority48173.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48200

namespace LeftBound48211
def owner : Owner := ⟨.program ⟨214⟩, ⟨17672⟩⟩
def transferEvent : Nat := 48211
def frameStart : Nat := 48123
def rule : BoundRule := .product (.predecessor 0 48209 .coefficient) (.predecessor 1 48210 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48209 .coefficient)
      LeftAuthority48184.bound (LeftAuthority48184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48184.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48210 .coefficient)
      LeftAuthority48207.bound (LeftAuthority48207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48207.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48207.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority48184.bound LeftAuthority48207.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48184.bound, LeftAuthority48207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority48184.actual selector witness) * (LeftAuthority48207.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48211

namespace LeftBound48219
def owner : Owner := ⟨.program ⟨214⟩, ⟨17673⟩⟩
def transferEvent : Nat := 48219
def frameStart : Nat := 48123
def rule : BoundRule := .sum [.predecessor 0 48217 .coefficient, .predecessor 1 48218 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48217 .coefficient)
      LeftAuthority48215.bound (LeftAuthority48215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48215.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48218 .coefficient)
      LeftBound48211.bound (LeftBound48211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48211.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48215.bound, LeftBound48211.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48215.bound, LeftBound48211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority48215.actual selector witness, LeftBound48211.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48219

namespace LeftBound48223
def owner : Owner := ⟨.program ⟨214⟩, ⟨28325⟩⟩
def transferEvent : Nat := 48223
def frameStart : Nat := 48123
def rule : BoundRule := .sum [.predecessor 0 48221 .coefficient, .predecessor 1 48222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48221 .coefficient)
      LeftBound48219.bound (LeftBound48219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48222 .coefficient)
      LeftBound48200.bound (LeftBound48200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48200.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48219.bound, LeftBound48200.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48219.bound, LeftBound48200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48219.actual selector witness, LeftBound48200.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48223

namespace LeftBound48236
def owner : Owner := ⟨.program ⟨214⟩, ⟨28322⟩⟩
def transferEvent : Nat := 48236
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 48234 .coefficient, .predecessor 1 48235 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48234 .coefficient)
      LeftBound48065.bound (LeftBound48065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48065.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48065.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48235 .coefficient)
      LeftBound48048.bound (LeftBound48048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48048.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48065.bound, LeftBound48048.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48065.bound, LeftBound48048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48065.actual selector witness, LeftBound48048.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48236

namespace LeftBound48239
def owner : Owner := ⟨.program ⟨214⟩, ⟨28322⟩⟩
def transferEvent : Nat := 48239
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 48233 .summary, .result 48055 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48233 .summary)
      LeftBound48067.bound (LeftBound48067.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21627⟩⟩) (rawTerms := some (Proof.Events188.exact48233RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48055 .summary)
      LeftBound48050.bound (LeftBound48050.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28321⟩⟩) (rawTerms := some (Proof.Events187.exact48055RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48067.bound, LeftBound48050.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48067.bound, LeftBound48050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48067.actual selector witness, LeftBound48050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48239

namespace LeftBound48243
def owner : Owner := ⟨.program ⟨214⟩, ⟨28323⟩⟩
def transferEvent : Nat := 48243
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48241 .coefficient) (.predecessor 1 48242 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48241 .coefficient)
      LeftBound48236.bound (LeftBound48236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48242 .coefficient)
      LeftBound5678.bound (LeftBound5678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48236.bound LeftBound5678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48236.bound, LeftBound5678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48236.actual selector witness) * (LeftBound5678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48243

namespace LeftBound48244
def owner : Owner := ⟨.program ⟨214⟩, ⟨28323⟩⟩
def transferEvent : Nat := 48244
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩ [⟨.result 5675 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5675 .coefficient)
      LeftAuthority5674.bound (LeftAuthority5674.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6681⟩⟩) (rawTerms := some (Proof.Events022.exact5675RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5674.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5674.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5674.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48244

namespace LeftBound48245
def owner : Owner := ⟨.program ⟨214⟩, ⟨28323⟩⟩
def transferEvent : Nat := 48245
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 48240 .summary) (.transfer 48244) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48240 .summary)
      LeftBound48239.bound (LeftBound48239.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28322⟩⟩) (rawTerms := some (Proof.Events188.exact48240RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48244)
      LeftBound48244.bound (LeftBound48244.actual selector witness) := by
  exact .transfer (LeftBound48244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48239.bound LeftBound48244.bound
def bound : CoeffClass := .finite ⟨4742323242612988221224648704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48239.bound, LeftBound48244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48239.actual selector witness) * (LeftBound48244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48245

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
