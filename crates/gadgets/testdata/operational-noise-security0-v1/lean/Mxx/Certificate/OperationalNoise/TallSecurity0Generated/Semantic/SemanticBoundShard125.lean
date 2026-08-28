import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard090
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard124

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound20280
def owner : Owner := ⟨.program ⟨214⟩, ⟨26828⟩⟩
def transferEvent : Nat := 20280
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26826⟩⟩]⟩ [⟨.result 20276 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20276 .coefficient)
      LeftAuthority20275.bound (LeftAuthority20275.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26826⟩⟩) (rawTerms := some (Proof.Events079.exact20276RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20275.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20275.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority20275.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20275.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20280

namespace LeftBound20281
def owner : Owner := ⟨.program ⟨214⟩, ⟨26828⟩⟩
def transferEvent : Nat := 20281
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 14262 .summary) (.transfer 20280) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14262 .summary)
      LeftBound14261.bound (LeftBound14261.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25087⟩⟩) (rawTerms := some (Proof.Events055.exact14262RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20280)
      LeftBound20280.bound (LeftBound20280.actual selector witness) := by
  exact .transfer (LeftBound20280.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14261.bound LeftBound20280.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14261.bound, LeftBound20280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14261.actual selector witness) * (LeftBound20280.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20281

namespace LeftBound20292
def owner : Owner := ⟨.program ⟨214⟩, ⟨20626⟩⟩
def transferEvent : Nat := 20292
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 20290 .coefficient) (.value (.predecessor 1 20291 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20290 .coefficient)
      LeftAuthority20288.bound (LeftAuthority20288.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20288.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20288.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20291 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority20288.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20288.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20288.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound20292

namespace LeftBound20296
def owner : Owner := ⟨.program ⟨214⟩, ⟨20627⟩⟩
def transferEvent : Nat := 20296
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20294 .coefficient) (.predecessor 1 20295 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20294 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20295 .coefficient)
      LeftBound20292.bound (LeftBound20292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20292.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20292.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound20292.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound20292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound20292.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20296

namespace LeftBound20297
def owner : Owner := ⟨.program ⟨214⟩, ⟨20627⟩⟩
def transferEvent : Nat := 20297
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20624⟩⟩]⟩ [⟨.result 20289 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20289 .coefficient)
      LeftAuthority20288.bound (LeftAuthority20288.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20624⟩⟩) (rawTerms := some (Proof.Events079.exact20289RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20288.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20288.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority20288.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20288.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20297

namespace LeftBound20298
def owner : Owner := ⟨.program ⟨214⟩, ⟨20627⟩⟩
def transferEvent : Nat := 20298
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 20297) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20297)
      LeftBound20297.bound (LeftBound20297.actual selector witness) := by
  exact .transfer (LeftBound20297.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound20297.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound20297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound20297.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20298

namespace LeftBound20393
def owner : Owner := ⟨.program ⟨214⟩, ⟨15131⟩⟩
def transferEvent : Nat := 20393
def frameStart : Nat := 20354
def rule : BoundRule := .identity (.predecessor 0 20392 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20392 .coefficient)
      LeftAuthority20390.bound (LeftAuthority20390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20390.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20390.derived selector witness)

def rawBound : CoeffClass := LeftAuthority20390.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority20390.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound20393

namespace LeftBound20410
def owner : Owner := ⟨.program ⟨214⟩, ⟨15170⟩⟩
def transferEvent : Nat := 20410
def frameStart : Nat := 20354
def rule : BoundRule := .sum [.predecessor 0 20408 .coefficient, .predecessor 1 20409 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20408 .coefficient)
      LeftBound20393.bound (LeftBound20393.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound20393.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20409 .coefficient)
      LeftAuthority20406.bound (LeftAuthority20406.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority20406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20393.bound, LeftAuthority20406.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20393.bound, LeftAuthority20406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20393.actual selector witness, LeftAuthority20406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20410

namespace LeftBound20413
def owner : Owner := ⟨.program ⟨214⟩, ⟨15171⟩⟩
def transferEvent : Nat := 20413
def frameStart : Nat := 20354
def rule : BoundRule := .identity (.predecessor 0 20412 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20412 .coefficient)
      LeftBound20410.bound (LeftBound20410.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound20410.derived selector witness)

def rawBound : CoeffClass := LeftBound20410.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound20410.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound20413

namespace LeftBound20419
def owner : Owner := ⟨.program ⟨214⟩, ⟨15172⟩⟩
def transferEvent : Nat := 20419
def frameStart : Nat := 20354
def rule : BoundRule := .product (.predecessor 0 20417 .coefficient) (.predecessor 1 20418 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20417 .coefficient)
      LeftAuthority20415.bound (LeftAuthority20415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20415.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20415.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20418 .coefficient)
      LeftBound20413.bound (LeftBound20413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20413.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20413.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority20415.bound LeftBound20413.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20415.bound, LeftBound20413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority20415.actual selector witness) * (LeftBound20413.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20419

namespace LeftBound20427
def owner : Owner := ⟨.program ⟨214⟩, ⟨15173⟩⟩
def transferEvent : Nat := 20427
def frameStart : Nat := 20354
def rule : BoundRule := .sum [.predecessor 0 20425 .coefficient, .predecessor 1 20426 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20425 .coefficient)
      LeftAuthority20423.bound (LeftAuthority20423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20423.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20423.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20426 .coefficient)
      LeftBound20419.bound (LeftBound20419.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20419.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20419.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority20423.bound, LeftBound20419.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20423.bound, LeftBound20419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority20423.actual selector witness, LeftBound20419.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20427

namespace LeftBound20431
def owner : Owner := ⟨.program ⟨214⟩, ⟨26827⟩⟩
def transferEvent : Nat := 20431
def frameStart : Nat := 20354
def rule : BoundRule := .product (.predecessor 0 20429 .coefficient) (.predecessor 1 20430 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20429 .coefficient)
      LeftBound20427.bound (LeftBound20427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20427.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20430 .coefficient)
      LeftAuthority20404.bound (LeftAuthority20404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20404.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20427.bound LeftAuthority20404.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20427.bound, LeftAuthority20404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20427.actual selector witness) * (LeftAuthority20404.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20431

namespace LeftBound20442
def owner : Owner := ⟨.program ⟨214⟩, ⟨15231⟩⟩
def transferEvent : Nat := 20442
def frameStart : Nat := 20354
def rule : BoundRule := .product (.predecessor 0 20440 .coefficient) (.predecessor 1 20441 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20440 .coefficient)
      LeftAuthority20415.bound (LeftAuthority20415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20415.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20415.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20441 .coefficient)
      LeftAuthority20438.bound (LeftAuthority20438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20438.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20438.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority20415.bound LeftAuthority20438.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20415.bound, LeftAuthority20438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority20415.actual selector witness) * (LeftAuthority20438.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20442

namespace LeftBound20450
def owner : Owner := ⟨.program ⟨214⟩, ⟨15232⟩⟩
def transferEvent : Nat := 20450
def frameStart : Nat := 20354
def rule : BoundRule := .sum [.predecessor 0 20448 .coefficient, .predecessor 1 20449 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20448 .coefficient)
      LeftAuthority20446.bound (LeftAuthority20446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20446.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20449 .coefficient)
      LeftBound20442.bound (LeftBound20442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority20446.bound, LeftBound20442.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20446.bound, LeftBound20442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority20446.actual selector witness, LeftBound20442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20450

namespace LeftBound20454
def owner : Owner := ⟨.program ⟨214⟩, ⟨26832⟩⟩
def transferEvent : Nat := 20454
def frameStart : Nat := 20354
def rule : BoundRule := .sum [.predecessor 0 20452 .coefficient, .predecessor 1 20453 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20452 .coefficient)
      LeftBound20450.bound (LeftBound20450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20450.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20453 .coefficient)
      LeftBound20431.bound (LeftBound20431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20431.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20450.bound, LeftBound20431.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20450.bound, LeftBound20431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20450.actual selector witness, LeftBound20431.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20454

namespace LeftBound20467
def owner : Owner := ⟨.program ⟨214⟩, ⟨26829⟩⟩
def transferEvent : Nat := 20467
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20465 .coefficient, .predecessor 1 20466 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20465 .coefficient)
      LeftBound20296.bound (LeftBound20296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20296.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20296.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20466 .coefficient)
      LeftBound20279.bound (LeftBound20279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20279.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20296.bound, LeftBound20279.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20296.bound, LeftBound20279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20296.actual selector witness, LeftBound20279.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20467

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
