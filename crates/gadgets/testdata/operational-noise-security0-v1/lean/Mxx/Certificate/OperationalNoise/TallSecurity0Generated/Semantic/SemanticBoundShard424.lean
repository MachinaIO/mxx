import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard381
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard423

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound63294
def owner : Owner := ⟨.program ⟨214⟩, ⟨27876⟩⟩
def transferEvent : Nat := 63294
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 63289 .summary) (.transfer 63293) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63289 .summary)
      LeftBound63288.bound (LeftBound63288.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27875⟩⟩) (rawTerms := some (Proof.Events247.exact63289RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63288.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63293)
      LeftBound63293.bound (LeftBound63293.actual selector witness) := by
  exact .transfer (LeftBound63293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63288.bound LeftBound63293.bound
def bound : CoeffClass := .finite ⟨4741911972453864866771369984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63288.bound, LeftBound63293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63288.actual selector witness) * (LeftBound63293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63294

namespace LeftBound63309
def owner : Owner := ⟨.program ⟨214⟩, ⟨27657⟩⟩
def transferEvent : Nat := 63309
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63307 .coefficient) (.predecessor 1 63308 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63307 .coefficient)
      LeftBound56246.bound (LeftBound56246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63308 .coefficient)
      LeftAuthority63305.bound (LeftAuthority63305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63305.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63305.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56246.bound LeftAuthority63305.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56246.bound, LeftAuthority63305.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56246.actual selector witness) * (LeftAuthority63305.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63309

namespace LeftBound63310
def owner : Owner := ⟨.program ⟨214⟩, ⟨27657⟩⟩
def transferEvent : Nat := 63310
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27655⟩⟩]⟩ [⟨.result 63306 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63306 .coefficient)
      LeftAuthority63305.bound (LeftAuthority63305.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27655⟩⟩) (rawTerms := some (Proof.Events247.exact63306RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63305.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63305.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63305.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63305.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63305.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63310

namespace LeftBound63311
def owner : Owner := ⟨.program ⟨214⟩, ⟨27657⟩⟩
def transferEvent : Nat := 63311
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 56250 .summary) (.transfer 63310) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56250 .summary)
      LeftBound56249.bound (LeftBound56249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25996⟩⟩) (rawTerms := some (Proof.Events219.exact56250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63310)
      LeftBound63310.bound (LeftBound63310.actual selector witness) := by
  exact .transfer (LeftBound63310.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56249.bound LeftBound63310.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56249.bound, LeftBound63310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56249.actual selector witness) * (LeftBound63310.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63311

namespace LeftBound63322
def owner : Owner := ⟨.program ⟨214⟩, ⟨21190⟩⟩
def transferEvent : Nat := 63322
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 63320 .coefficient) (.value (.predecessor 1 63321 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63320 .coefficient)
      LeftAuthority63318.bound (LeftAuthority63318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63321 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority63318.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63318.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63318.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound63322

namespace LeftBound63326
def owner : Owner := ⟨.program ⟨214⟩, ⟨21191⟩⟩
def transferEvent : Nat := 63326
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63324 .coefficient) (.predecessor 1 63325 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63324 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63325 .coefficient)
      LeftBound63322.bound (LeftBound63322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63322.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound63322.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound63322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound63322.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63326

namespace LeftBound63327
def owner : Owner := ⟨.program ⟨214⟩, ⟨21191⟩⟩
def transferEvent : Nat := 63327
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21188⟩⟩]⟩ [⟨.result 63319 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63319 .coefficient)
      LeftAuthority63318.bound (LeftAuthority63318.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21188⟩⟩) (rawTerms := some (Proof.Events247.exact63319RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63318.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63318.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63318.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63327

namespace LeftBound63328
def owner : Owner := ⟨.program ⟨214⟩, ⟨21191⟩⟩
def transferEvent : Nat := 63328
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 63327) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63327)
      LeftBound63327.bound (LeftBound63327.actual selector witness) := by
  exact .transfer (LeftBound63327.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound63327.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound63327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound63327.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63328

namespace LeftBound63423
def owner : Owner := ⟨.program ⟨214⟩, ⟨15826⟩⟩
def transferEvent : Nat := 63423
def frameStart : Nat := 63384
def rule : BoundRule := .identity (.predecessor 0 63422 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63422 .coefficient)
      LeftAuthority63420.bound (LeftAuthority63420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63420.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63420.derived selector witness)

def rawBound : CoeffClass := LeftAuthority63420.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority63420.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63423

namespace LeftBound63440
def owner : Owner := ⟨.program ⟨214⟩, ⟨15900⟩⟩
def transferEvent : Nat := 63440
def frameStart : Nat := 63384
def rule : BoundRule := .sum [.predecessor 0 63438 .coefficient, .predecessor 1 63439 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63438 .coefficient)
      LeftBound63423.bound (LeftBound63423.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63423.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63439 .coefficient)
      LeftAuthority63436.bound (LeftAuthority63436.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority63436.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63423.bound, LeftAuthority63436.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63423.bound, LeftAuthority63436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63423.actual selector witness, LeftAuthority63436.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63440

namespace LeftBound63443
def owner : Owner := ⟨.program ⟨214⟩, ⟨15901⟩⟩
def transferEvent : Nat := 63443
def frameStart : Nat := 63384
def rule : BoundRule := .identity (.predecessor 0 63442 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63442 .coefficient)
      LeftBound63440.bound (LeftBound63440.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63440.derived selector witness)

def rawBound : CoeffClass := LeftBound63440.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound63440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63443

namespace LeftBound63449
def owner : Owner := ⟨.program ⟨214⟩, ⟨15902⟩⟩
def transferEvent : Nat := 63449
def frameStart : Nat := 63384
def rule : BoundRule := .product (.predecessor 0 63447 .coefficient) (.predecessor 1 63448 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63447 .coefficient)
      LeftAuthority63445.bound (LeftAuthority63445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63445.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63445.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63448 .coefficient)
      LeftBound63443.bound (LeftBound63443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63443.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority63445.bound LeftBound63443.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63445.bound, LeftBound63443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority63445.actual selector witness) * (LeftBound63443.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63449

namespace LeftBound63457
def owner : Owner := ⟨.program ⟨214⟩, ⟨15903⟩⟩
def transferEvent : Nat := 63457
def frameStart : Nat := 63384
def rule : BoundRule := .sum [.predecessor 0 63455 .coefficient, .predecessor 1 63456 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63455 .coefficient)
      LeftAuthority63453.bound (LeftAuthority63453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63453.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63453.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63456 .coefficient)
      LeftBound63449.bound (LeftBound63449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63449.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63449.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63453.bound, LeftBound63449.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63453.bound, LeftBound63449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63453.actual selector witness, LeftBound63449.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63457

namespace LeftBound63461
def owner : Owner := ⟨.program ⟨214⟩, ⟨27656⟩⟩
def transferEvent : Nat := 63461
def frameStart : Nat := 63384
def rule : BoundRule := .product (.predecessor 0 63459 .coefficient) (.predecessor 1 63460 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63459 .coefficient)
      LeftBound63457.bound (LeftBound63457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63457.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63460 .coefficient)
      LeftAuthority63434.bound (LeftAuthority63434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63434.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63434.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63457.bound LeftAuthority63434.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63457.bound, LeftAuthority63434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63457.actual selector witness) * (LeftAuthority63434.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63461

namespace LeftBound63472
def owner : Owner := ⟨.program ⟨214⟩, ⟨17227⟩⟩
def transferEvent : Nat := 63472
def frameStart : Nat := 63384
def rule : BoundRule := .product (.predecessor 0 63470 .coefficient) (.predecessor 1 63471 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63470 .coefficient)
      LeftAuthority63445.bound (LeftAuthority63445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63445.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63445.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63471 .coefficient)
      LeftAuthority63468.bound (LeftAuthority63468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63468.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority63445.bound LeftAuthority63468.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63445.bound, LeftAuthority63468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority63445.actual selector witness) * (LeftAuthority63468.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63472

namespace LeftBound63480
def owner : Owner := ⟨.program ⟨214⟩, ⟨17228⟩⟩
def transferEvent : Nat := 63480
def frameStart : Nat := 63384
def rule : BoundRule := .sum [.predecessor 0 63478 .coefficient, .predecessor 1 63479 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63478 .coefficient)
      LeftAuthority63476.bound (LeftAuthority63476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63476.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63476.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63479 .coefficient)
      LeftBound63472.bound (LeftBound63472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63472.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63476.bound, LeftBound63472.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63476.bound, LeftBound63472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63476.actual selector witness, LeftBound63472.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63480

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
