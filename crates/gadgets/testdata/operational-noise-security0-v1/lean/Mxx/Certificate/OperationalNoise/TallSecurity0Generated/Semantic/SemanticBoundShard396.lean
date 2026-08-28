import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard091
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard395

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound58199
def owner : Owner := ⟨.program ⟨214⟩, ⟨20687⟩⟩
def transferEvent : Nat := 58199
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20684⟩⟩]⟩ [⟨.result 58191 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58191 .coefficient)
      LeftAuthority58190.bound (LeftAuthority58190.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20684⟩⟩) (rawTerms := some (Proof.Events227.exact58191RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58190.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority58190.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority58190.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58199

namespace LeftBound58200
def owner : Owner := ⟨.program ⟨214⟩, ⟨20687⟩⟩
def transferEvent : Nat := 58200
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 58199) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 58199)
      LeftBound58199.bound (LeftBound58199.actual selector witness) := by
  exact .transfer (LeftBound58199.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound58199.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound58199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound58199.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58200

namespace LeftBound58295
def owner : Owner := ⟨.program ⟨214⟩, ⟨15119⟩⟩
def transferEvent : Nat := 58295
def frameStart : Nat := 58256
def rule : BoundRule := .identity (.predecessor 0 58294 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58294 .coefficient)
      LeftAuthority58292.bound (LeftAuthority58292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58292.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58292.derived selector witness)

def rawBound : CoeffClass := LeftAuthority58292.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority58292.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound58295

namespace LeftBound58312
def owner : Owner := ⟨.program ⟨214⟩, ⟨15158⟩⟩
def transferEvent : Nat := 58312
def frameStart : Nat := 58256
def rule : BoundRule := .sum [.predecessor 0 58310 .coefficient, .predecessor 1 58311 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58310 .coefficient)
      LeftBound58295.bound (LeftBound58295.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound58295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58311 .coefficient)
      LeftAuthority58308.bound (LeftAuthority58308.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority58308.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58295.bound, LeftAuthority58308.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58295.bound, LeftAuthority58308.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58295.actual selector witness, LeftAuthority58308.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58312

namespace LeftBound58315
def owner : Owner := ⟨.program ⟨214⟩, ⟨15159⟩⟩
def transferEvent : Nat := 58315
def frameStart : Nat := 58256
def rule : BoundRule := .identity (.predecessor 0 58314 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58314 .coefficient)
      LeftBound58312.bound (LeftBound58312.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound58312.derived selector witness)

def rawBound : CoeffClass := LeftBound58312.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound58312.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound58315

namespace LeftBound58321
def owner : Owner := ⟨.program ⟨214⟩, ⟨15160⟩⟩
def transferEvent : Nat := 58321
def frameStart : Nat := 58256
def rule : BoundRule := .product (.predecessor 0 58319 .coefficient) (.predecessor 1 58320 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58319 .coefficient)
      LeftAuthority58317.bound (LeftAuthority58317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58317.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58320 .coefficient)
      LeftBound58315.bound (LeftBound58315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58315.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority58317.bound LeftBound58315.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58317.bound, LeftBound58315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority58317.actual selector witness) * (LeftBound58315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58321

namespace LeftBound58329
def owner : Owner := ⟨.program ⟨214⟩, ⟨15161⟩⟩
def transferEvent : Nat := 58329
def frameStart : Nat := 58256
def rule : BoundRule := .sum [.predecessor 0 58327 .coefficient, .predecessor 1 58328 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58327 .coefficient)
      LeftAuthority58325.bound (LeftAuthority58325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58325.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58328 .coefficient)
      LeftBound58321.bound (LeftBound58321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58321.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority58325.bound, LeftBound58321.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58325.bound, LeftBound58321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority58325.actual selector witness, LeftBound58321.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58329

namespace LeftBound58333
def owner : Owner := ⟨.program ⟨214⟩, ⟨26795⟩⟩
def transferEvent : Nat := 58333
def frameStart : Nat := 58256
def rule : BoundRule := .product (.predecessor 0 58331 .coefficient) (.predecessor 1 58332 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58331 .coefficient)
      LeftBound58329.bound (LeftBound58329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58329.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58332 .coefficient)
      LeftAuthority58306.bound (LeftAuthority58306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58329.bound LeftAuthority58306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58329.bound, LeftAuthority58306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58329.actual selector witness) * (LeftAuthority58306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58333

namespace LeftBound58344
def owner : Owner := ⟨.program ⟨214⟩, ⟨15372⟩⟩
def transferEvent : Nat := 58344
def frameStart : Nat := 58256
def rule : BoundRule := .product (.predecessor 0 58342 .coefficient) (.predecessor 1 58343 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58342 .coefficient)
      LeftAuthority58317.bound (LeftAuthority58317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58317.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58343 .coefficient)
      LeftAuthority58340.bound (LeftAuthority58340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58340.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58340.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority58317.bound LeftAuthority58340.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58317.bound, LeftAuthority58340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority58317.actual selector witness) * (LeftAuthority58340.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58344

namespace LeftBound58352
def owner : Owner := ⟨.program ⟨214⟩, ⟨15373⟩⟩
def transferEvent : Nat := 58352
def frameStart : Nat := 58256
def rule : BoundRule := .sum [.predecessor 0 58350 .coefficient, .predecessor 1 58351 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58350 .coefficient)
      LeftAuthority58348.bound (LeftAuthority58348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58351 .coefficient)
      LeftBound58344.bound (LeftBound58344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58344.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58344.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority58348.bound, LeftBound58344.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58348.bound, LeftBound58344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority58348.actual selector witness, LeftBound58344.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58352

namespace LeftBound58356
def owner : Owner := ⟨.program ⟨214⟩, ⟨26799⟩⟩
def transferEvent : Nat := 58356
def frameStart : Nat := 58256
def rule : BoundRule := .sum [.predecessor 0 58354 .coefficient, .predecessor 1 58355 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58354 .coefficient)
      LeftBound58352.bound (LeftBound58352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58355 .coefficient)
      LeftBound58333.bound (LeftBound58333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58333.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58352.bound, LeftBound58333.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58352.bound, LeftBound58333.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58352.actual selector witness, LeftBound58333.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58356

namespace LeftBound58369
def owner : Owner := ⟨.program ⟨214⟩, ⟨26797⟩⟩
def transferEvent : Nat := 58369
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58367 .coefficient, .predecessor 1 58368 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58367 .coefficient)
      LeftBound58198.bound (LeftBound58198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58368 .coefficient)
      LeftBound58181.bound (LeftBound58181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58181.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58198.bound, LeftBound58181.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58198.bound, LeftBound58181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58198.actual selector witness, LeftBound58181.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58369

namespace LeftBound58372
def owner : Owner := ⟨.program ⟨214⟩, ⟨26797⟩⟩
def transferEvent : Nat := 58372
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 58366 .summary, .result 58188 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58366 .summary)
      LeftBound58200.bound (LeftBound58200.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20687⟩⟩) (rawTerms := some (Proof.Events227.exact58366RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58188 .summary)
      LeftBound58183.bound (LeftBound58183.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26796⟩⟩) (rawTerms := some (Proof.Events227.exact58188RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58183.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58200.bound, LeftBound58183.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58200.bound, LeftBound58183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58200.actual selector witness, LeftBound58183.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58372

namespace LeftBound58396
def owner : Owner := ⟨.program ⟨214⟩, ⟨10687⟩⟩
def transferEvent : Nat := 58396
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 58394 .coefficient) (.predecessor 1 58395 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58394 .coefficient)
      LeftAuthority2705.bound (LeftAuthority2705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2705.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2705.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58395 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2705.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2705.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2705.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound58396

namespace LeftBound58401
def owner : Owner := ⟨.program ⟨214⟩, ⟨7267⟩⟩
def transferEvent : Nat := 58401
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58399 .coefficient) (.predecessor 1 58400 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58399 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58400 .coefficient)
      LeftBound14487.bound (LeftBound14487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14487.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound14487.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound14487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound14487.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58401

namespace LeftBound58406
def owner : Owner := ⟨.program ⟨214⟩, ⟨10688⟩⟩
def transferEvent : Nat := 58406
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58404 .coefficient, .predecessor 1 58405 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58404 .coefficient)
      LeftBound58401.bound (LeftBound58401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58405 .coefficient)
      LeftBound58396.bound (LeftBound58396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58396.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58396.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58401.bound, LeftBound58396.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58401.bound, LeftBound58396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58401.actual selector witness, LeftBound58396.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58406

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
