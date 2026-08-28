import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard215

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound33212
def owner : Owner := ⟨.program ⟨214⟩, ⟨28551⟩⟩
def transferEvent : Nat := 33212
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28549⟩⟩]⟩ [⟨.result 33208 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33208 .coefficient)
      LeftAuthority33207.bound (LeftAuthority33207.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28549⟩⟩) (rawTerms := some (Proof.Events129.exact33208RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33207.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33207.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority33207.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33207.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33212

namespace LeftBound33213
def owner : Owner := ⟨.program ⟨214⟩, ⟨28551⟩⟩
def transferEvent : Nat := 33213
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 25072 .summary) (.transfer 33212) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25072 .summary)
      LeftBound25071.bound (LeftBound25071.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25159⟩⟩) (rawTerms := some (Proof.Events097.exact25072RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33212)
      LeftBound33212.bound (LeftBound33212.actual selector witness) := by
  exact .transfer (LeftBound33212.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25071.bound LeftBound33212.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25071.bound, LeftBound33212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25071.actual selector witness) * (LeftBound33212.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33213

namespace LeftBound33224
def owner : Owner := ⟨.program ⟨214⟩, ⟨21774⟩⟩
def transferEvent : Nat := 33224
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 33222 .coefficient) (.value (.predecessor 1 33223 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33222 .coefficient)
      LeftAuthority33220.bound (LeftAuthority33220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33220.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33220.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33223 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority33220.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33220.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33220.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound33224

namespace LeftBound33228
def owner : Owner := ⟨.program ⟨214⟩, ⟨21775⟩⟩
def transferEvent : Nat := 33228
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33226 .coefficient) (.predecessor 1 33227 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33226 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33227 .coefficient)
      LeftBound33224.bound (LeftBound33224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33224.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound33224.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound33224.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound33224.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33228

namespace LeftBound33229
def owner : Owner := ⟨.program ⟨214⟩, ⟨21775⟩⟩
def transferEvent : Nat := 33229
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21772⟩⟩]⟩ [⟨.result 33221 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33221 .coefficient)
      LeftAuthority33220.bound (LeftAuthority33220.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21772⟩⟩) (rawTerms := some (Proof.Events129.exact33221RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33220.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33220.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority33220.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33220.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33229

namespace LeftBound33230
def owner : Owner := ⟨.program ⟨214⟩, ⟨21775⟩⟩
def transferEvent : Nat := 33230
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 33229) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33229)
      LeftBound33229.bound (LeftBound33229.actual selector witness) := by
  exact .transfer (LeftBound33229.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound33229.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound33229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound33229.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33230

namespace LeftBound33325
def owner : Owner := ⟨.program ⟨214⟩, ⟨16275⟩⟩
def transferEvent : Nat := 33325
def frameStart : Nat := 33286
def rule : BoundRule := .identity (.predecessor 0 33324 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33324 .coefficient)
      LeftAuthority33322.bound (LeftAuthority33322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33322.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33322.derived selector witness)

def rawBound : CoeffClass := LeftAuthority33322.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority33322.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound33325

namespace LeftBound33342
def owner : Owner := ⟨.program ⟨214⟩, ⟨16349⟩⟩
def transferEvent : Nat := 33342
def frameStart : Nat := 33286
def rule : BoundRule := .sum [.predecessor 0 33340 .coefficient, .predecessor 1 33341 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33340 .coefficient)
      LeftBound33325.bound (LeftBound33325.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound33325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33341 .coefficient)
      LeftAuthority33338.bound (LeftAuthority33338.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority33338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33325.bound, LeftAuthority33338.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33325.bound, LeftAuthority33338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33325.actual selector witness, LeftAuthority33338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33342

namespace LeftBound33345
def owner : Owner := ⟨.program ⟨214⟩, ⟨16350⟩⟩
def transferEvent : Nat := 33345
def frameStart : Nat := 33286
def rule : BoundRule := .identity (.predecessor 0 33344 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33344 .coefficient)
      LeftBound33342.bound (LeftBound33342.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound33342.derived selector witness)

def rawBound : CoeffClass := LeftBound33342.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound33342.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound33345

namespace LeftBound33351
def owner : Owner := ⟨.program ⟨214⟩, ⟨16351⟩⟩
def transferEvent : Nat := 33351
def frameStart : Nat := 33286
def rule : BoundRule := .product (.predecessor 0 33349 .coefficient) (.predecessor 1 33350 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33349 .coefficient)
      LeftAuthority33347.bound (LeftAuthority33347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33347.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33350 .coefficient)
      LeftBound33345.bound (LeftBound33345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33345.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority33347.bound LeftBound33345.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33347.bound, LeftBound33345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority33347.actual selector witness) * (LeftBound33345.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33351

namespace LeftBound33359
def owner : Owner := ⟨.program ⟨214⟩, ⟨16352⟩⟩
def transferEvent : Nat := 33359
def frameStart : Nat := 33286
def rule : BoundRule := .sum [.predecessor 0 33357 .coefficient, .predecessor 1 33358 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33357 .coefficient)
      LeftAuthority33355.bound (LeftAuthority33355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33355.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33358 .coefficient)
      LeftBound33351.bound (LeftBound33351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33351.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority33355.bound, LeftBound33351.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33355.bound, LeftBound33351.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority33355.actual selector witness, LeftBound33351.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33359

namespace LeftBound33363
def owner : Owner := ⟨.program ⟨214⟩, ⟨28550⟩⟩
def transferEvent : Nat := 33363
def frameStart : Nat := 33286
def rule : BoundRule := .product (.predecessor 0 33361 .coefficient) (.predecessor 1 33362 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33361 .coefficient)
      LeftBound33359.bound (LeftBound33359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33359.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33362 .coefficient)
      LeftAuthority33336.bound (LeftAuthority33336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33336.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33336.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33359.bound LeftAuthority33336.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33359.bound, LeftAuthority33336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33359.actual selector witness) * (LeftAuthority33336.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33363

namespace LeftBound33374
def owner : Owner := ⟨.program ⟨214⟩, ⟨17620⟩⟩
def transferEvent : Nat := 33374
def frameStart : Nat := 33286
def rule : BoundRule := .product (.predecessor 0 33372 .coefficient) (.predecessor 1 33373 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33372 .coefficient)
      LeftAuthority33347.bound (LeftAuthority33347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33347.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33373 .coefficient)
      LeftAuthority33370.bound (LeftAuthority33370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33370.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority33347.bound LeftAuthority33370.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33347.bound, LeftAuthority33370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority33347.actual selector witness) * (LeftAuthority33370.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33374

namespace LeftBound33382
def owner : Owner := ⟨.program ⟨214⟩, ⟨17621⟩⟩
def transferEvent : Nat := 33382
def frameStart : Nat := 33286
def rule : BoundRule := .sum [.predecessor 0 33380 .coefficient, .predecessor 1 33381 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33380 .coefficient)
      LeftAuthority33378.bound (LeftAuthority33378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33381 .coefficient)
      LeftBound33374.bound (LeftBound33374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33374.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33374.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority33378.bound, LeftBound33374.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33378.bound, LeftBound33374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority33378.actual selector witness, LeftBound33374.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33382

namespace LeftBound33386
def owner : Owner := ⟨.program ⟨214⟩, ⟨28555⟩⟩
def transferEvent : Nat := 33386
def frameStart : Nat := 33286
def rule : BoundRule := .sum [.predecessor 0 33384 .coefficient, .predecessor 1 33385 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33384 .coefficient)
      LeftBound33382.bound (LeftBound33382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33382.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33382.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33385 .coefficient)
      LeftBound33363.bound (LeftBound33363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33363.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33363.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33382.bound, LeftBound33363.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33382.bound, LeftBound33363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33382.actual selector witness, LeftBound33363.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33386

namespace LeftBound33399
def owner : Owner := ⟨.program ⟨214⟩, ⟨28552⟩⟩
def transferEvent : Nat := 33399
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 33397 .coefficient, .predecessor 1 33398 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33397 .coefficient)
      LeftBound33228.bound (LeftBound33228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33398 .coefficient)
      LeftBound33211.bound (LeftBound33211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33211.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33228.bound, LeftBound33211.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33228.bound, LeftBound33211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33228.actual selector witness, LeftBound33211.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33399

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
