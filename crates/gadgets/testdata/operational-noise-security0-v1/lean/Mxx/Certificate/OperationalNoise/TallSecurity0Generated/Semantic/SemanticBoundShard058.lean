import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound10067
def owner : Owner := ⟨.program ⟨214⟩, ⟨19762⟩⟩
def transferEvent : Nat := 10067
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 10065 .coefficient) (.value (.predecessor 1 10066 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10065 .coefficient)
      LeftAuthority10063.bound (LeftAuthority10063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10063.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10066 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority10063.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10063.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10063.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound10067

namespace LeftBound10071
def owner : Owner := ⟨.program ⟨214⟩, ⟨19763⟩⟩
def transferEvent : Nat := 10071
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10069 .coefficient) (.predecessor 1 10070 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10069 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10070 .coefficient)
      LeftBound10067.bound (LeftBound10067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10067.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound10067.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound10067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound10067.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10071

namespace LeftBound10072
def owner : Owner := ⟨.program ⟨214⟩, ⟨19763⟩⟩
def transferEvent : Nat := 10072
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19760⟩⟩]⟩ [⟨.result 10064 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10064 .coefficient)
      LeftAuthority10063.bound (LeftAuthority10063.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19760⟩⟩) (rawTerms := some (Proof.Events039.exact10064RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10063.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10063.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10063.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10063.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10072

namespace LeftBound10073
def owner : Owner := ⟨.program ⟨214⟩, ⟨19763⟩⟩
def transferEvent : Nat := 10073
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 10072) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10072)
      LeftBound10072.bound (LeftBound10072.actual selector witness) := by
  exact .transfer (LeftBound10072.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound10072.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound10072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound10072.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10073

namespace LeftBound10152
def owner : Owner := ⟨.program ⟨214⟩, ⟨11794⟩⟩
def transferEvent : Nat := 10152
def frameStart : Nat := 10123
def rule : BoundRule := .product (.predecessor 0 10150 .coefficient) (.predecessor 1 10151 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10150 .coefficient)
      LeftAuthority10148.bound (LeftAuthority10148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10148.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10148.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10151 .coefficient)
      LeftAuthority10145.bound (LeftAuthority10145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10145.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10145.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority10148.bound LeftAuthority10145.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10148.bound, LeftAuthority10145.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority10148.actual selector witness) * (LeftAuthority10145.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10152

namespace LeftBound10156
def owner : Owner := ⟨.program ⟨214⟩, ⟨11795⟩⟩
def transferEvent : Nat := 10156
def frameStart : Nat := 10123
def rule : BoundRule := .identity (.predecessor 0 10155 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10155 .coefficient)
      LeftBound10152.bound (LeftBound10152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10152.derived selector witness)

def rawBound : CoeffClass := LeftBound10152.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound10152.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10156

namespace LeftBound10173
def owner : Owner := ⟨.program ⟨214⟩, ⟨11873⟩⟩
def transferEvent : Nat := 10173
def frameStart : Nat := 10123
def rule : BoundRule := .sum [.predecessor 0 10171 .coefficient, .predecessor 1 10172 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10171 .coefficient)
      LeftBound10156.bound (LeftBound10156.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound10156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10172 .coefficient)
      LeftAuthority10169.bound (LeftAuthority10169.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority10169.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10156.bound, LeftAuthority10169.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10156.bound, LeftAuthority10169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10156.actual selector witness, LeftAuthority10169.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10173

namespace LeftBound10176
def owner : Owner := ⟨.program ⟨214⟩, ⟨11874⟩⟩
def transferEvent : Nat := 10176
def frameStart : Nat := 10123
def rule : BoundRule := .identity (.predecessor 0 10175 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10175 .coefficient)
      LeftBound10173.bound (LeftBound10173.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound10173.derived selector witness)

def rawBound : CoeffClass := LeftBound10173.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound10173.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10176

namespace LeftBound10182
def owner : Owner := ⟨.program ⟨214⟩, ⟨11875⟩⟩
def transferEvent : Nat := 10182
def frameStart : Nat := 10123
def rule : BoundRule := .product (.predecessor 0 10180 .coefficient) (.predecessor 1 10181 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10180 .coefficient)
      LeftAuthority10178.bound (LeftAuthority10178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10178.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10181 .coefficient)
      LeftBound10176.bound (LeftBound10176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10176.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority10178.bound LeftBound10176.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10178.bound, LeftBound10176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority10178.actual selector witness) * (LeftBound10176.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10182

namespace LeftBound10198
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def transferEvent : Nat := 10198
def frameStart : Nat := 10123
def rule : BoundRule := .scale (.predecessor 0 10196 .coefficient) (.value (.predecessor 1 10197 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10196 .coefficient)
      LeftAuthority10194.bound (LeftAuthority10194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10194.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10197 .coefficient)
      LeftAuthority10185.bound (LeftAuthority10185.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority10185.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority10194.bound LeftAuthority10185.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10194.bound, LeftAuthority10185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10194.actual selector witness) * (LeftAuthority10185.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound10198

namespace LeftBound10201
def owner : Owner := ⟨.program ⟨214⟩, ⟨6763⟩⟩
def transferEvent : Nat := 10201
def frameStart : Nat := 10123
def rule : BoundRule := .identity (.predecessor 0 10200 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10200 .coefficient)
      LeftAuthority10188.bound (LeftAuthority10188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10188.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10188.derived selector witness)

def rawBound : CoeffClass := LeftAuthority10188.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10188.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority10188.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10201

namespace LeftBound10205
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def transferEvent : Nat := 10205
def frameStart : Nat := 10123
def rule : BoundRule := .product (.predecessor 0 10203 .coefficient) (.predecessor 1 10204 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10203 .coefficient)
      LeftBound10201.bound (LeftBound10201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10204 .coefficient)
      LeftBound10198.bound (LeftBound10198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10198.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10201.bound LeftBound10198.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10201.bound, LeftBound10198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10201.actual selector witness) * (LeftBound10198.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10205

namespace LeftBound10210
def owner : Owner := ⟨.program ⟨214⟩, ⟨11876⟩⟩
def transferEvent : Nat := 10210
def frameStart : Nat := 10123
def rule : BoundRule := .sum [.predecessor 0 10208 .coefficient, .predecessor 1 10209 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10208 .coefficient)
      LeftBound10205.bound (LeftBound10205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10205.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10209 .coefficient)
      LeftBound10182.bound (LeftBound10182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10182.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10205.bound, LeftBound10182.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10205.bound, LeftBound10182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10205.actual selector witness, LeftBound10182.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10210

namespace LeftBound10214
def owner : Owner := ⟨.program ⟨214⟩, ⟨25165⟩⟩
def transferEvent : Nat := 10214
def frameStart : Nat := 10123
def rule : BoundRule := .product (.predecessor 0 10212 .coefficient) (.predecessor 1 10213 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10212 .coefficient)
      LeftBound10210.bound (LeftBound10210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10213 .coefficient)
      LeftAuthority10167.bound (LeftAuthority10167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10167.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10210.bound LeftAuthority10167.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10210.bound, LeftAuthority10167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10210.actual selector witness) * (LeftAuthority10167.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10214

namespace LeftBound10225
def owner : Owner := ⟨.program ⟨214⟩, ⟨16280⟩⟩
def transferEvent : Nat := 10225
def frameStart : Nat := 10123
def rule : BoundRule := .product (.predecessor 0 10223 .coefficient) (.predecessor 1 10224 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10223 .coefficient)
      LeftAuthority10178.bound (LeftAuthority10178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10178.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10224 .coefficient)
      LeftAuthority10221.bound (LeftAuthority10221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10221.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10221.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority10178.bound LeftAuthority10221.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10178.bound, LeftAuthority10221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority10178.actual selector witness) * (LeftAuthority10221.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10225

namespace LeftBound10233
def owner : Owner := ⟨.program ⟨214⟩, ⟨16281⟩⟩
def transferEvent : Nat := 10233
def frameStart : Nat := 10123
def rule : BoundRule := .sum [.predecessor 0 10231 .coefficient, .predecessor 1 10232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10231 .coefficient)
      LeftAuthority10229.bound (LeftAuthority10229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10229.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10232 .coefficient)
      LeftBound10225.bound (LeftBound10225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10225.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10225.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority10229.bound, LeftBound10225.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10229.bound, LeftBound10225.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority10229.actual selector witness, LeftBound10225.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10233

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
