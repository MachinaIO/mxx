import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard484

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound71159
def owner : Owner := ⟨.program ⟨214⟩, ⟨25908⟩⟩
def transferEvent : Nat := 71159
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 71154 .summary) (.transfer 71158) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71154 .summary)
      LeftBound71153.bound (LeftBound71153.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13772⟩⟩) (rawTerms := some (Proof.Events277.exact71154RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 71158)
      LeftBound71158.bound (LeftBound71158.actual selector witness) := by
  exact .transfer (LeftBound71158.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71153.bound LeftBound71158.bound
def bound : CoeffClass := .finite ⟨350231094886400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71153.bound, LeftBound71158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71153.actual selector witness) * (LeftBound71158.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71159

namespace LeftBound71170
def owner : Owner := ⟨.program ⟨214⟩, ⟨19382⟩⟩
def transferEvent : Nat := 71170
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 71168 .coefficient) (.value (.predecessor 1 71169 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71168 .coefficient)
      LeftAuthority71166.bound (LeftAuthority71166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71169 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority71166.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71166.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71166.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound71170

namespace LeftBound71174
def owner : Owner := ⟨.program ⟨214⟩, ⟨19383⟩⟩
def transferEvent : Nat := 71174
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71172 .coefficient) (.predecessor 1 71173 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71172 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71173 .coefficient)
      LeftBound71170.bound (LeftBound71170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71170.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound71170.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound71170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound71170.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71174

namespace LeftBound71175
def owner : Owner := ⟨.program ⟨214⟩, ⟨19383⟩⟩
def transferEvent : Nat := 71175
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩ [⟨.result 71167 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71167 .coefficient)
      LeftAuthority71166.bound (LeftAuthority71166.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19380⟩⟩) (rawTerms := some (Proof.Events277.exact71167RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71166.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority71166.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71166.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71175

namespace LeftBound71176
def owner : Owner := ⟨.program ⟨214⟩, ⟨19383⟩⟩
def transferEvent : Nat := 71176
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 71175) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 71175)
      LeftBound71175.bound (LeftBound71175.actual selector witness) := by
  exact .transfer (LeftBound71175.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound71175.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound71175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound71175.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71176

namespace LeftBound71255
def owner : Owner := ⟨.program ⟨214⟩, ⟨13765⟩⟩
def transferEvent : Nat := 71255
def frameStart : Nat := 71226
def rule : BoundRule := .product (.predecessor 0 71253 .coefficient) (.predecessor 1 71254 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71253 .coefficient)
      LeftAuthority71251.bound (LeftAuthority71251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71251.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71254 .coefficient)
      LeftAuthority71248.bound (LeftAuthority71248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71248.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71248.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority71251.bound LeftAuthority71248.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71251.bound, LeftAuthority71248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority71251.actual selector witness) * (LeftAuthority71248.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71255

namespace LeftBound71259
def owner : Owner := ⟨.program ⟨214⟩, ⟨13766⟩⟩
def transferEvent : Nat := 71259
def frameStart : Nat := 71226
def rule : BoundRule := .identity (.predecessor 0 71258 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71258 .coefficient)
      LeftBound71255.bound (LeftBound71255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71255.derived selector witness)

def rawBound : CoeffClass := LeftBound71255.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound71255.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71259

namespace LeftBound71276
def owner : Owner := ⟨.program ⟨214⟩, ⟨13876⟩⟩
def transferEvent : Nat := 71276
def frameStart : Nat := 71226
def rule : BoundRule := .sum [.predecessor 0 71274 .coefficient, .predecessor 1 71275 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71274 .coefficient)
      LeftBound71259.bound (LeftBound71259.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71275 .coefficient)
      LeftAuthority71272.bound (LeftAuthority71272.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority71272.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71259.bound, LeftAuthority71272.bound]
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71259.bound, LeftAuthority71272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71259.actual selector witness, LeftAuthority71272.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71276

namespace LeftBound71279
def owner : Owner := ⟨.program ⟨214⟩, ⟨13877⟩⟩
def transferEvent : Nat := 71279
def frameStart : Nat := 71226
def rule : BoundRule := .identity (.predecessor 0 71278 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71278 .coefficient)
      LeftBound71276.bound (LeftBound71276.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71276.derived selector witness)

def rawBound : CoeffClass := LeftBound71276.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound71276.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71279

namespace LeftBound71285
def owner : Owner := ⟨.program ⟨214⟩, ⟨13878⟩⟩
def transferEvent : Nat := 71285
def frameStart : Nat := 71226
def rule : BoundRule := .product (.predecessor 0 71283 .coefficient) (.predecessor 1 71284 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71283 .coefficient)
      LeftAuthority71281.bound (LeftAuthority71281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71281.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71284 .coefficient)
      LeftBound71279.bound (LeftBound71279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71279.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority71281.bound LeftBound71279.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71281.bound, LeftBound71279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority71281.actual selector witness) * (LeftBound71279.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71285

namespace LeftBound71301
def owner : Owner := ⟨.program ⟨214⟩, ⟨7847⟩⟩
def transferEvent : Nat := 71301
def frameStart : Nat := 71226
def rule : BoundRule := .scale (.predecessor 0 71299 .coefficient) (.value (.predecessor 1 71300 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71299 .coefficient)
      LeftAuthority71297.bound (LeftAuthority71297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71297.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71300 .coefficient)
      LeftAuthority71288.bound (LeftAuthority71288.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority71288.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority71297.bound LeftAuthority71288.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71297.bound, LeftAuthority71288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71297.actual selector witness) * (LeftAuthority71288.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound71301

namespace LeftBound71304
def owner : Owner := ⟨.program ⟨214⟩, ⟨6794⟩⟩
def transferEvent : Nat := 71304
def frameStart : Nat := 71226
def rule : BoundRule := .identity (.predecessor 0 71303 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71303 .coefficient)
      LeftAuthority71291.bound (LeftAuthority71291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71291.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71291.derived selector witness)

def rawBound : CoeffClass := LeftAuthority71291.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority71291.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71304

namespace LeftBound71308
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def transferEvent : Nat := 71308
def frameStart : Nat := 71226
def rule : BoundRule := .product (.predecessor 0 71306 .coefficient) (.predecessor 1 71307 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71306 .coefficient)
      LeftBound71304.bound (LeftBound71304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71304.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71304.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71307 .coefficient)
      LeftBound71301.bound (LeftBound71301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71301.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71304.bound LeftBound71301.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71304.bound, LeftBound71301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71304.actual selector witness) * (LeftBound71301.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71308

namespace LeftBound71313
def owner : Owner := ⟨.program ⟨214⟩, ⟨13879⟩⟩
def transferEvent : Nat := 71313
def frameStart : Nat := 71226
def rule : BoundRule := .sum [.predecessor 0 71311 .coefficient, .predecessor 1 71312 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71311 .coefficient)
      LeftBound71308.bound (LeftBound71308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71308.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71312 .coefficient)
      LeftBound71285.bound (LeftBound71285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71285.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71308.bound, LeftBound71285.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71308.bound, LeftBound71285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71308.actual selector witness, LeftBound71285.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71313

namespace LeftBound71317
def owner : Owner := ⟨.program ⟨214⟩, ⟨25910⟩⟩
def transferEvent : Nat := 71317
def frameStart : Nat := 71226
def rule : BoundRule := .product (.predecessor 0 71315 .coefficient) (.predecessor 1 71316 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71315 .coefficient)
      LeftBound71313.bound (LeftBound71313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71316 .coefficient)
      LeftAuthority71270.bound (LeftAuthority71270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71270.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71313.bound LeftAuthority71270.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71313.bound, LeftAuthority71270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71313.actual selector witness) * (LeftAuthority71270.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71317

namespace LeftBound71328
def owner : Owner := ⟨.program ⟨214⟩, ⟨15700⟩⟩
def transferEvent : Nat := 71328
def frameStart : Nat := 71226
def rule : BoundRule := .product (.predecessor 0 71326 .coefficient) (.predecessor 1 71327 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71326 .coefficient)
      LeftAuthority71281.bound (LeftAuthority71281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71281.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71327 .coefficient)
      LeftAuthority71324.bound (LeftAuthority71324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority71281.bound LeftAuthority71324.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71281.bound, LeftAuthority71324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority71281.actual selector witness) * (LeftAuthority71324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71328

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
