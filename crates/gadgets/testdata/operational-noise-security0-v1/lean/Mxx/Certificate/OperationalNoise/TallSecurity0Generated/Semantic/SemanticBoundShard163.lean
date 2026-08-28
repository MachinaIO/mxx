import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard162

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound24991
def owner : Owner := ⟨.program ⟨214⟩, ⟨11869⟩⟩
def transferEvent : Nat := 24991
def frameStart : Nat := 24941
def rule : BoundRule := .sum [.predecessor 0 24989 .coefficient, .predecessor 1 24990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24989 .coefficient)
      LeftBound24974.bound (LeftBound24974.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24990 .coefficient)
      LeftAuthority24987.bound (LeftAuthority24987.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24974.bound, LeftAuthority24987.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24974.bound, LeftAuthority24987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24974.actual selector witness, LeftAuthority24987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24991

namespace LeftBound24994
def owner : Owner := ⟨.program ⟨214⟩, ⟨11870⟩⟩
def transferEvent : Nat := 24994
def frameStart : Nat := 24941
def rule : BoundRule := .identity (.predecessor 0 24993 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24993 .coefficient)
      LeftBound24991.bound (LeftBound24991.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24991.derived selector witness)

def rawBound : CoeffClass := LeftBound24991.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24991.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound24991.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24994

namespace LeftBound25000
def owner : Owner := ⟨.program ⟨214⟩, ⟨11871⟩⟩
def transferEvent : Nat := 25000
def frameStart : Nat := 24941
def rule : BoundRule := .product (.predecessor 0 24998 .coefficient) (.predecessor 1 24999 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24998 .coefficient)
      LeftAuthority24996.bound (LeftAuthority24996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24999 .coefficient)
      LeftBound24994.bound (LeftBound24994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24994.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority24996.bound LeftBound24994.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24996.bound, LeftBound24994.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority24996.actual selector witness) * (LeftBound24994.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25000

namespace LeftBound25016
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def transferEvent : Nat := 25016
def frameStart : Nat := 24941
def rule : BoundRule := .scale (.predecessor 0 25014 .coefficient) (.value (.predecessor 1 25015 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25014 .coefficient)
      LeftAuthority25012.bound (LeftAuthority25012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25015 .coefficient)
      LeftAuthority25003.bound (LeftAuthority25003.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority25003.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority25012.bound LeftAuthority25003.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25012.bound, LeftAuthority25003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25012.actual selector witness) * (LeftAuthority25003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25016

namespace LeftBound25019
def owner : Owner := ⟨.program ⟨214⟩, ⟨6763⟩⟩
def transferEvent : Nat := 25019
def frameStart : Nat := 24941
def rule : BoundRule := .identity (.predecessor 0 25018 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25018 .coefficient)
      LeftAuthority25006.bound (LeftAuthority25006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25006.derived selector witness)

def rawBound : CoeffClass := LeftAuthority25006.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority25006.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25019

namespace LeftBound25023
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def transferEvent : Nat := 25023
def frameStart : Nat := 24941
def rule : BoundRule := .product (.predecessor 0 25021 .coefficient) (.predecessor 1 25022 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25021 .coefficient)
      LeftBound25019.bound (LeftBound25019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25022 .coefficient)
      LeftBound25016.bound (LeftBound25016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25019.bound LeftBound25016.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25019.bound, LeftBound25016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25019.actual selector witness) * (LeftBound25016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25023

namespace LeftBound25028
def owner : Owner := ⟨.program ⟨214⟩, ⟨11872⟩⟩
def transferEvent : Nat := 25028
def frameStart : Nat := 24941
def rule : BoundRule := .sum [.predecessor 0 25026 .coefficient, .predecessor 1 25027 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25026 .coefficient)
      LeftBound25023.bound (LeftBound25023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25027 .coefficient)
      LeftBound25000.bound (LeftBound25000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25000.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25023.bound, LeftBound25000.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25023.bound, LeftBound25000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25023.actual selector witness, LeftBound25000.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25028

namespace LeftBound25032
def owner : Owner := ⟨.program ⟨214⟩, ⟨25160⟩⟩
def transferEvent : Nat := 25032
def frameStart : Nat := 24941
def rule : BoundRule := .product (.predecessor 0 25030 .coefficient) (.predecessor 1 25031 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25030 .coefficient)
      LeftBound25028.bound (LeftBound25028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25031 .coefficient)
      LeftAuthority24985.bound (LeftAuthority24985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24985.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24985.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25028.bound LeftAuthority24985.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25028.bound, LeftAuthority24985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25028.actual selector witness) * (LeftAuthority24985.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25032

namespace LeftBound25043
def owner : Owner := ⟨.program ⟨214⟩, ⟨16276⟩⟩
def transferEvent : Nat := 25043
def frameStart : Nat := 24941
def rule : BoundRule := .product (.predecessor 0 25041 .coefficient) (.predecessor 1 25042 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25041 .coefficient)
      LeftAuthority24996.bound (LeftAuthority24996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25042 .coefficient)
      LeftAuthority25039.bound (LeftAuthority25039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25039.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24996.bound LeftAuthority25039.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24996.bound, LeftAuthority25039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority24996.actual selector witness) * (LeftAuthority25039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25043

namespace LeftBound25051
def owner : Owner := ⟨.program ⟨214⟩, ⟨16277⟩⟩
def transferEvent : Nat := 25051
def frameStart : Nat := 24941
def rule : BoundRule := .sum [.predecessor 0 25049 .coefficient, .predecessor 1 25050 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25049 .coefficient)
      LeftAuthority25047.bound (LeftAuthority25047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25047.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25050 .coefficient)
      LeftBound25043.bound (LeftBound25043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25043.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority25047.bound, LeftBound25043.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25047.bound, LeftBound25043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority25047.actual selector witness, LeftBound25043.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25051

namespace LeftBound25055
def owner : Owner := ⟨.program ⟨214⟩, ⟨25161⟩⟩
def transferEvent : Nat := 25055
def frameStart : Nat := 24941
def rule : BoundRule := .sum [.predecessor 0 25053 .coefficient, .predecessor 1 25054 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25053 .coefficient)
      LeftBound25051.bound (LeftBound25051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25054 .coefficient)
      LeftBound25032.bound (LeftBound25032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25032.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25051.bound, LeftBound25032.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25051.bound, LeftBound25032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25051.actual selector witness, LeftBound25032.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25055

namespace LeftBound25068
def owner : Owner := ⟨.program ⟨214⟩, ⟨25159⟩⟩
def transferEvent : Nat := 25068
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25066 .coefficient, .predecessor 1 25067 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25066 .coefficient)
      LeftBound24889.bound (LeftBound24889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25067 .coefficient)
      LeftBound24872.bound (LeftBound24872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24889.bound, LeftBound24872.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24889.bound, LeftBound24872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24889.actual selector witness, LeftBound24872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25068

namespace LeftBound25071
def owner : Owner := ⟨.program ⟨214⟩, ⟨25159⟩⟩
def transferEvent : Nat := 25071
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 25065 .summary, .result 24879 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25065 .summary)
      LeftBound24891.bound (LeftBound24891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19759⟩⟩) (rawTerms := some (Proof.Events097.exact25065RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24879 .summary)
      LeftBound24874.bound (LeftBound24874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25158⟩⟩) (rawTerms := some (Proof.Events097.exact24879RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24891.bound, LeftBound24874.bound]
def bound : CoeffClass := .finite ⟨352097360556032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24891.bound, LeftBound24874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24891.actual selector witness, LeftBound24874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25071

namespace LeftBound25075
def owner : Owner := ⟨.program ⟨214⟩, ⟨28558⟩⟩
def transferEvent : Nat := 25075
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25073 .coefficient) (.predecessor 1 25074 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25073 .coefficient)
      LeftBound25068.bound (LeftBound25068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25074 .coefficient)
      LeftAuthority24794.bound (LeftAuthority24794.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24794.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25068.bound LeftAuthority24794.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25068.bound, LeftAuthority24794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25068.actual selector witness) * (LeftAuthority24794.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25075

namespace LeftBound25076
def owner : Owner := ⟨.program ⟨214⟩, ⟨28558⟩⟩
def transferEvent : Nat := 25076
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28556⟩⟩]⟩ [⟨.result 24795 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24795 .coefficient)
      LeftAuthority24794.bound (LeftAuthority24794.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28556⟩⟩) (rawTerms := some (Proof.Events096.exact24795RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24794.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority24794.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24794.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25076

namespace LeftBound25077
def owner : Owner := ⟨.program ⟨214⟩, ⟨28558⟩⟩
def transferEvent : Nat := 25077
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 25072 .summary) (.transfer 25076) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25072 .summary)
      LeftBound25071.bound (LeftBound25071.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25159⟩⟩) (rawTerms := some (Proof.Events097.exact25072RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 25076)
      LeftBound25076.bound (LeftBound25076.actual selector witness) := by
  exact .transfer (LeftBound25076.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25071.bound LeftBound25076.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25071.bound, LeftBound25076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25071.actual selector witness) * (LeftBound25076.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25077

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
