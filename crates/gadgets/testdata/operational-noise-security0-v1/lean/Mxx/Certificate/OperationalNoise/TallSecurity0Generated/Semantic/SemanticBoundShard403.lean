import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard402

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound59146
def owner : Owner := ⟨.program ⟨214⟩, ⟨26372⟩⟩
def transferEvent : Nat := 59146
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26370⟩⟩]⟩ [⟨.result 58865 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58865 .coefficient)
      LeftAuthority58864.bound (LeftAuthority58864.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26370⟩⟩) (rawTerms := some (Proof.Events229.exact58865RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58864.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority58864.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority58864.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound59146

namespace LeftBound59147
def owner : Owner := ⟨.program ⟨214⟩, ⟨26372⟩⟩
def transferEvent : Nat := 59147
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 59142 .summary) (.transfer 59146) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59142 .summary)
      LeftBound59141.bound (LeftBound59141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24918⟩⟩) (rawTerms := some (Proof.Events231.exact59142RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 59146)
      LeftBound59146.bound (LeftBound59146.actual selector witness) := by
  exact .transfer (LeftBound59146.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound59141.bound LeftBound59146.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59141.bound, LeftBound59146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound59141.actual selector witness) * (LeftBound59146.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59147

namespace LeftBound59158
def owner : Owner := ⟨.program ⟨214⟩, ⟨20398⟩⟩
def transferEvent : Nat := 59158
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 59156 .coefficient) (.value (.predecessor 1 59157 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59156 .coefficient)
      LeftAuthority59154.bound (LeftAuthority59154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59157 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority59154.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59154.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority59154.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound59158

namespace LeftBound59162
def owner : Owner := ⟨.program ⟨214⟩, ⟨20399⟩⟩
def transferEvent : Nat := 59162
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 59160 .coefficient) (.predecessor 1 59161 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59160 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59161 .coefficient)
      LeftBound59158.bound (LeftBound59158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59158.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound59158.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound59158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound59158.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59162

namespace LeftBound59163
def owner : Owner := ⟨.program ⟨214⟩, ⟨20399⟩⟩
def transferEvent : Nat := 59163
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20396⟩⟩]⟩ [⟨.result 59155 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59155 .coefficient)
      LeftAuthority59154.bound (LeftAuthority59154.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20396⟩⟩) (rawTerms := some (Proof.Events231.exact59155RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59154.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority59154.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59154.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority59154.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound59163

namespace LeftBound59164
def owner : Owner := ⟨.program ⟨214⟩, ⟨20399⟩⟩
def transferEvent : Nat := 59164
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 59163) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 59163)
      LeftBound59163.bound (LeftBound59163.actual selector witness) := by
  exact .transfer (LeftBound59163.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound59163.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound59163.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound59163.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59164

namespace LeftBound59259
def owner : Owner := ⟨.program ⟨214⟩, ⟨14797⟩⟩
def transferEvent : Nat := 59259
def frameStart : Nat := 59220
def rule : BoundRule := .identity (.predecessor 0 59258 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59258 .coefficient)
      LeftAuthority59256.bound (LeftAuthority59256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59256.derived selector witness)

def rawBound : CoeffClass := LeftAuthority59256.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority59256.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound59259

namespace LeftBound59276
def owner : Owner := ⟨.program ⟨214⟩, ⟨14836⟩⟩
def transferEvent : Nat := 59276
def frameStart : Nat := 59220
def rule : BoundRule := .sum [.predecessor 0 59274 .coefficient, .predecessor 1 59275 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59274 .coefficient)
      LeftBound59259.bound (LeftBound59259.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound59259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59275 .coefficient)
      LeftAuthority59272.bound (LeftAuthority59272.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority59272.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59259.bound, LeftAuthority59272.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59259.bound, LeftAuthority59272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59259.actual selector witness, LeftAuthority59272.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59276

namespace LeftBound59279
def owner : Owner := ⟨.program ⟨214⟩, ⟨14837⟩⟩
def transferEvent : Nat := 59279
def frameStart : Nat := 59220
def rule : BoundRule := .identity (.predecessor 0 59278 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59278 .coefficient)
      LeftBound59276.bound (LeftBound59276.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound59276.derived selector witness)

def rawBound : CoeffClass := LeftBound59276.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound59276.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound59279

namespace LeftBound59285
def owner : Owner := ⟨.program ⟨214⟩, ⟨14838⟩⟩
def transferEvent : Nat := 59285
def frameStart : Nat := 59220
def rule : BoundRule := .product (.predecessor 0 59283 .coefficient) (.predecessor 1 59284 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59283 .coefficient)
      LeftAuthority59281.bound (LeftAuthority59281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59281.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59284 .coefficient)
      LeftBound59279.bound (LeftBound59279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59279.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority59281.bound LeftBound59279.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59281.bound, LeftBound59279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority59281.actual selector witness) * (LeftBound59279.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59285

namespace LeftBound59293
def owner : Owner := ⟨.program ⟨214⟩, ⟨14839⟩⟩
def transferEvent : Nat := 59293
def frameStart : Nat := 59220
def rule : BoundRule := .sum [.predecessor 0 59291 .coefficient, .predecessor 1 59292 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59291 .coefficient)
      LeftAuthority59289.bound (LeftAuthority59289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59292 .coefficient)
      LeftBound59285.bound (LeftBound59285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59285.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority59289.bound, LeftBound59285.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59289.bound, LeftBound59285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority59289.actual selector witness, LeftBound59285.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59293

namespace LeftBound59297
def owner : Owner := ⟨.program ⟨214⟩, ⟨26371⟩⟩
def transferEvent : Nat := 59297
def frameStart : Nat := 59220
def rule : BoundRule := .product (.predecessor 0 59295 .coefficient) (.predecessor 1 59296 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59295 .coefficient)
      LeftBound59293.bound (LeftBound59293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59296 .coefficient)
      LeftAuthority59270.bound (LeftAuthority59270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59270.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound59293.bound LeftAuthority59270.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59293.bound, LeftAuthority59270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound59293.actual selector witness) * (LeftAuthority59270.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59297

namespace LeftBound59308
def owner : Owner := ⟨.program ⟨214⟩, ⟨15269⟩⟩
def transferEvent : Nat := 59308
def frameStart : Nat := 59220
def rule : BoundRule := .product (.predecessor 0 59306 .coefficient) (.predecessor 1 59307 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59306 .coefficient)
      LeftAuthority59281.bound (LeftAuthority59281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59281.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59307 .coefficient)
      LeftAuthority59304.bound (LeftAuthority59304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59304.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59304.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority59281.bound LeftAuthority59304.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59281.bound, LeftAuthority59304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority59281.actual selector witness) * (LeftAuthority59304.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59308

namespace LeftBound59316
def owner : Owner := ⟨.program ⟨214⟩, ⟨15270⟩⟩
def transferEvent : Nat := 59316
def frameStart : Nat := 59220
def rule : BoundRule := .sum [.predecessor 0 59314 .coefficient, .predecessor 1 59315 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59314 .coefficient)
      LeftAuthority59312.bound (LeftAuthority59312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59315 .coefficient)
      LeftBound59308.bound (LeftBound59308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59308.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59308.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority59312.bound, LeftBound59308.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59312.bound, LeftBound59308.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority59312.actual selector witness, LeftBound59308.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59316

namespace LeftBound59320
def owner : Owner := ⟨.program ⟨214⟩, ⟨26374⟩⟩
def transferEvent : Nat := 59320
def frameStart : Nat := 59220
def rule : BoundRule := .sum [.predecessor 0 59318 .coefficient, .predecessor 1 59319 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59318 .coefficient)
      LeftBound59316.bound (LeftBound59316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59319 .coefficient)
      LeftBound59297.bound (LeftBound59297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59297.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59316.bound, LeftBound59297.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59316.bound, LeftBound59297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59316.actual selector witness, LeftBound59297.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59320

namespace LeftBound59333
def owner : Owner := ⟨.program ⟨214⟩, ⟨26373⟩⟩
def transferEvent : Nat := 59333
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59331 .coefficient, .predecessor 1 59332 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59331 .coefficient)
      LeftBound59162.bound (LeftBound59162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59332 .coefficient)
      LeftBound59145.bound (LeftBound59145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59145.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59145.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59162.bound, LeftBound59145.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59162.bound, LeftBound59145.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59162.actual selector witness, LeftBound59145.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59333

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
