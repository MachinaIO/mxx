import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard252

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound38051
def owner : Owner := ⟨.program ⟨214⟩, ⟨25461⟩⟩
def transferEvent : Nat := 38051
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38049 .coefficient) (.predecessor 1 38050 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38049 .coefficient)
      LeftBound38045.bound (LeftBound38045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38045.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38050 .coefficient)
      LeftAuthority37983.bound (LeftAuthority37983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37983.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38045.bound LeftAuthority37983.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38045.bound, LeftAuthority37983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38045.actual selector witness) * (LeftAuthority37983.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38051

namespace LeftBound38052
def owner : Owner := ⟨.program ⟨214⟩, ⟨25461⟩⟩
def transferEvent : Nat := 38052
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25460⟩⟩]⟩ [⟨.result 37984 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37984 .coefficient)
      LeftAuthority37983.bound (LeftAuthority37983.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25460⟩⟩) (rawTerms := some (Proof.Events148.exact37984RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37983.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37983.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37983.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38052

namespace LeftBound38053
def owner : Owner := ⟨.program ⟨214⟩, ⟨25461⟩⟩
def transferEvent : Nat := 38053
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38048 .summary) (.transfer 38052) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38048 .summary)
      LeftBound38047.bound (LeftBound38047.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12589⟩⟩) (rawTerms := some (Proof.Events148.exact38048RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38052)
      LeftBound38052.bound (LeftBound38052.actual selector witness) := by
  exact .transfer (LeftBound38052.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38047.bound LeftBound38052.bound
def bound : CoeffClass := .finite ⟨350322698485760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38047.bound, LeftBound38052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38047.actual selector witness) * (LeftBound38052.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38053

namespace LeftBound38064
def owner : Owner := ⟨.program ⟨214⟩, ⟨19970⟩⟩
def transferEvent : Nat := 38064
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 38062 .coefficient) (.value (.predecessor 1 38063 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38062 .coefficient)
      LeftAuthority38060.bound (LeftAuthority38060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38060.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38060.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38063 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority38060.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38060.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38060.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38064

namespace LeftBound38068
def owner : Owner := ⟨.program ⟨214⟩, ⟨19971⟩⟩
def transferEvent : Nat := 38068
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38066 .coefficient) (.predecessor 1 38067 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38066 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38067 .coefficient)
      LeftBound38064.bound (LeftBound38064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38064.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38064.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound38064.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound38064.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound38064.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38068

namespace LeftBound38069
def owner : Owner := ⟨.program ⟨214⟩, ⟨19971⟩⟩
def transferEvent : Nat := 38069
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19968⟩⟩]⟩ [⟨.result 38061 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38061 .coefficient)
      LeftAuthority38060.bound (LeftAuthority38060.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19968⟩⟩) (rawTerms := some (Proof.Events148.exact38061RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38060.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38060.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority38060.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38060.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38069

namespace LeftBound38070
def owner : Owner := ⟨.program ⟨214⟩, ⟨19971⟩⟩
def transferEvent : Nat := 38070
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 38069) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38069)
      LeftBound38069.bound (LeftBound38069.actual selector witness) := by
  exact .transfer (LeftBound38069.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound38069.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound38069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound38069.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38070

namespace LeftBound38149
def owner : Owner := ⟨.program ⟨214⟩, ⟨12583⟩⟩
def transferEvent : Nat := 38149
def frameStart : Nat := 38120
def rule : BoundRule := .product (.predecessor 0 38147 .coefficient) (.predecessor 1 38148 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38147 .coefficient)
      LeftAuthority38145.bound (LeftAuthority38145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38145.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38148 .coefficient)
      LeftAuthority38142.bound (LeftAuthority38142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38142.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38142.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority38145.bound LeftAuthority38142.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38145.bound, LeftAuthority38142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority38145.actual selector witness) * (LeftAuthority38142.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38149

namespace LeftBound38153
def owner : Owner := ⟨.program ⟨214⟩, ⟨12584⟩⟩
def transferEvent : Nat := 38153
def frameStart : Nat := 38120
def rule : BoundRule := .identity (.predecessor 0 38152 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38152 .coefficient)
      LeftBound38149.bound (LeftBound38149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38149.derived selector witness)

def rawBound : CoeffClass := LeftBound38149.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound38149.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38153

namespace LeftBound38170
def owner : Owner := ⟨.program ⟨214⟩, ⟨12670⟩⟩
def transferEvent : Nat := 38170
def frameStart : Nat := 38120
def rule : BoundRule := .sum [.predecessor 0 38168 .coefficient, .predecessor 1 38169 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38168 .coefficient)
      LeftBound38153.bound (LeftBound38153.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38169 .coefficient)
      LeftAuthority38166.bound (LeftAuthority38166.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority38166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38153.bound, LeftAuthority38166.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38153.bound, LeftAuthority38166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38153.actual selector witness, LeftAuthority38166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38170

namespace LeftBound38173
def owner : Owner := ⟨.program ⟨214⟩, ⟨12671⟩⟩
def transferEvent : Nat := 38173
def frameStart : Nat := 38120
def rule : BoundRule := .identity (.predecessor 0 38172 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38172 .coefficient)
      LeftBound38170.bound (LeftBound38170.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38170.derived selector witness)

def rawBound : CoeffClass := LeftBound38170.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound38170.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38173

namespace LeftBound38179
def owner : Owner := ⟨.program ⟨214⟩, ⟨12672⟩⟩
def transferEvent : Nat := 38179
def frameStart : Nat := 38120
def rule : BoundRule := .product (.predecessor 0 38177 .coefficient) (.predecessor 1 38178 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38177 .coefficient)
      LeftAuthority38175.bound (LeftAuthority38175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38178 .coefficient)
      LeftBound38173.bound (LeftBound38173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38173.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority38175.bound LeftBound38173.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38175.bound, LeftBound38173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority38175.actual selector witness) * (LeftBound38173.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38179

namespace LeftBound38195
def owner : Owner := ⟨.program ⟨214⟩, ⟨7871⟩⟩
def transferEvent : Nat := 38195
def frameStart : Nat := 38120
def rule : BoundRule := .scale (.predecessor 0 38193 .coefficient) (.value (.predecessor 1 38194 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38193 .coefficient)
      LeftAuthority38191.bound (LeftAuthority38191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38191.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38194 .coefficient)
      LeftAuthority38182.bound (LeftAuthority38182.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority38182.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority38191.bound LeftAuthority38182.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38191.bound, LeftAuthority38182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38191.actual selector witness) * (LeftAuthority38182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38195

namespace LeftBound38198
def owner : Owner := ⟨.program ⟨214⟩, ⟨6766⟩⟩
def transferEvent : Nat := 38198
def frameStart : Nat := 38120
def rule : BoundRule := .identity (.predecessor 0 38197 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38197 .coefficient)
      LeftAuthority38185.bound (LeftAuthority38185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38185.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38185.derived selector witness)

def rawBound : CoeffClass := LeftAuthority38185.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority38185.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38198

namespace LeftBound38202
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def transferEvent : Nat := 38202
def frameStart : Nat := 38120
def rule : BoundRule := .product (.predecessor 0 38200 .coefficient) (.predecessor 1 38201 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38200 .coefficient)
      LeftBound38198.bound (LeftBound38198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38201 .coefficient)
      LeftBound38195.bound (LeftBound38195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38195.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38195.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38198.bound LeftBound38195.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38198.bound, LeftBound38195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38198.actual selector witness) * (LeftBound38195.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38202

namespace LeftBound38207
def owner : Owner := ⟨.program ⟨214⟩, ⟨12673⟩⟩
def transferEvent : Nat := 38207
def frameStart : Nat := 38120
def rule : BoundRule := .sum [.predecessor 0 38205 .coefficient, .predecessor 1 38206 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38205 .coefficient)
      LeftBound38202.bound (LeftBound38202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38206 .coefficient)
      LeftBound38179.bound (LeftBound38179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38202.bound, LeftBound38179.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38202.bound, LeftBound38179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38202.actual selector witness, LeftBound38179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38207

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
