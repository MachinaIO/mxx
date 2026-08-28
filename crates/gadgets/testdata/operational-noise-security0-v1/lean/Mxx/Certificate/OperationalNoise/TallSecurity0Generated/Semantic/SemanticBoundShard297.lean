import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard296

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound43937
def owner : Owner := ⟨.program ⟨214⟩, ⟨10694⟩⟩
def transferEvent : Nat := 43937
def frameStart : Nat := 43904
def rule : BoundRule := .identity (.predecessor 0 43936 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43936 .coefficient)
      LeftBound43933.bound (LeftBound43933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43933.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43933.derived selector witness)

def rawBound : CoeffClass := LeftBound43933.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound43933.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43937

namespace LeftBound43954
def owner : Owner := ⟨.program ⟨214⟩, ⟨10780⟩⟩
def transferEvent : Nat := 43954
def frameStart : Nat := 43904
def rule : BoundRule := .sum [.predecessor 0 43952 .coefficient, .predecessor 1 43953 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43952 .coefficient)
      LeftBound43937.bound (LeftBound43937.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43953 .coefficient)
      LeftAuthority43950.bound (LeftAuthority43950.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority43950.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43937.bound, LeftAuthority43950.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43937.bound, LeftAuthority43950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43937.actual selector witness, LeftAuthority43950.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43954

namespace LeftBound43957
def owner : Owner := ⟨.program ⟨214⟩, ⟨10781⟩⟩
def transferEvent : Nat := 43957
def frameStart : Nat := 43904
def rule : BoundRule := .identity (.predecessor 0 43956 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43956 .coefficient)
      LeftBound43954.bound (LeftBound43954.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43954.derived selector witness)

def rawBound : CoeffClass := LeftBound43954.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound43954.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43957

namespace LeftBound43963
def owner : Owner := ⟨.program ⟨214⟩, ⟨10782⟩⟩
def transferEvent : Nat := 43963
def frameStart : Nat := 43904
def rule : BoundRule := .product (.predecessor 0 43961 .coefficient) (.predecessor 1 43962 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43961 .coefficient)
      LeftAuthority43959.bound (LeftAuthority43959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43959.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43962 .coefficient)
      LeftBound43957.bound (LeftBound43957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43957.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43957.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority43959.bound LeftBound43957.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43959.bound, LeftBound43957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority43959.actual selector witness) * (LeftBound43957.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43963

namespace LeftBound43979
def owner : Owner := ⟨.program ⟨214⟩, ⟨7835⟩⟩
def transferEvent : Nat := 43979
def frameStart : Nat := 43904
def rule : BoundRule := .scale (.predecessor 0 43977 .coefficient) (.value (.predecessor 1 43978 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43977 .coefficient)
      LeftAuthority43975.bound (LeftAuthority43975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43975.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43978 .coefficient)
      LeftAuthority43966.bound (LeftAuthority43966.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority43966.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority43975.bound LeftAuthority43966.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43975.bound, LeftAuthority43966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43975.actual selector witness) * (LeftAuthority43966.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound43979

namespace LeftBound43982
def owner : Owner := ⟨.program ⟨214⟩, ⟨6782⟩⟩
def transferEvent : Nat := 43982
def frameStart : Nat := 43904
def rule : BoundRule := .identity (.predecessor 0 43981 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43981 .coefficient)
      LeftAuthority43969.bound (LeftAuthority43969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43969.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43969.derived selector witness)

def rawBound : CoeffClass := LeftAuthority43969.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority43969.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43982

namespace LeftBound43986
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def transferEvent : Nat := 43986
def frameStart : Nat := 43904
def rule : BoundRule := .product (.predecessor 0 43984 .coefficient) (.predecessor 1 43985 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43984 .coefficient)
      LeftBound43982.bound (LeftBound43982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43985 .coefficient)
      LeftBound43979.bound (LeftBound43979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43979.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43982.bound LeftBound43979.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43982.bound, LeftBound43979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43982.actual selector witness) * (LeftBound43979.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43986

namespace LeftBound43991
def owner : Owner := ⟨.program ⟨214⟩, ⟨10783⟩⟩
def transferEvent : Nat := 43991
def frameStart : Nat := 43904
def rule : BoundRule := .sum [.predecessor 0 43989 .coefficient, .predecessor 1 43990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43989 .coefficient)
      LeftBound43986.bound (LeftBound43986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43990 .coefficient)
      LeftBound43963.bound (LeftBound43963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43986.bound, LeftBound43963.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43986.bound, LeftBound43963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43986.actual selector witness, LeftBound43963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43991

namespace LeftBound43995
def owner : Owner := ⟨.program ⟨214⟩, ⟨25001⟩⟩
def transferEvent : Nat := 43995
def frameStart : Nat := 43904
def rule : BoundRule := .product (.predecessor 0 43993 .coefficient) (.predecessor 1 43994 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43993 .coefficient)
      LeftBound43991.bound (LeftBound43991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43994 .coefficient)
      LeftAuthority43948.bound (LeftAuthority43948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43948.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43948.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43991.bound LeftAuthority43948.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43991.bound, LeftAuthority43948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43991.actual selector witness) * (LeftAuthority43948.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43995

namespace LeftBound44006
def owner : Owner := ⟨.program ⟨214⟩, ⟨14963⟩⟩
def transferEvent : Nat := 44006
def frameStart : Nat := 43904
def rule : BoundRule := .product (.predecessor 0 44004 .coefficient) (.predecessor 1 44005 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44004 .coefficient)
      LeftAuthority43959.bound (LeftAuthority43959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43959.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44005 .coefficient)
      LeftAuthority44002.bound (LeftAuthority44002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact44003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority43959.bound LeftAuthority44002.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43959.bound, LeftAuthority44002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority43959.actual selector witness) * (LeftAuthority44002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44006

namespace LeftBound44014
def owner : Owner := ⟨.program ⟨214⟩, ⟨14964⟩⟩
def transferEvent : Nat := 44014
def frameStart : Nat := 43904
def rule : BoundRule := .sum [.predecessor 0 44012 .coefficient, .predecessor 1 44013 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44012 .coefficient)
      LeftAuthority44010.bound (LeftAuthority44010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact44011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44013 .coefficient)
      LeftBound44006.bound (LeftBound44006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact44008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority44010.bound, LeftBound44006.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44010.bound, LeftBound44006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority44010.actual selector witness, LeftBound44006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44014

namespace LeftBound44018
def owner : Owner := ⟨.program ⟨214⟩, ⟨25002⟩⟩
def transferEvent : Nat := 44018
def frameStart : Nat := 43904
def rule : BoundRule := .sum [.predecessor 0 44016 .coefficient, .predecessor 1 44017 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44016 .coefficient)
      LeftBound44014.bound (LeftBound44014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact44015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44017 .coefficient)
      LeftBound43995.bound (LeftBound43995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact44000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44014.bound, LeftBound43995.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44014.bound, LeftBound43995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44014.actual selector witness, LeftBound43995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44018

namespace LeftBound44031
def owner : Owner := ⟨.program ⟨214⟩, ⟨25000⟩⟩
def transferEvent : Nat := 44031
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44029 .coefficient, .predecessor 1 44030 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44029 .coefficient)
      LeftBound43852.bound (LeftBound43852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact44028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44030 .coefficient)
      LeftBound43835.bound (LeftBound43835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43852.bound, LeftBound43835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43852.bound, LeftBound43835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43852.actual selector witness, LeftBound43835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44031

namespace LeftBound44034
def owner : Owner := ⟨.program ⟨214⟩, ⟨25000⟩⟩
def transferEvent : Nat := 44034
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44028 .summary, .result 43842 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44028 .summary)
      LeftBound43854.bound (LeftBound43854.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19107⟩⟩) (rawTerms := some (Proof.Events171.exact44028RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43842 .summary)
      LeftBound43837.bound (LeftBound43837.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24999⟩⟩) (rawTerms := some (Proof.Events171.exact43842RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43837.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43854.bound, LeftBound43837.bound]
def bound : CoeffClass := .finite ⟨352014917316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43854.bound, LeftBound43837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43854.actual selector witness, LeftBound43837.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44034

namespace LeftBound44038
def owner : Owner := ⟨.program ⟨214⟩, ⟨26592⟩⟩
def transferEvent : Nat := 44038
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44036 .coefficient) (.predecessor 1 44037 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44036 .coefficient)
      LeftBound44031.bound (LeftBound44031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44037 .coefficient)
      LeftAuthority43757.bound (LeftAuthority43757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43758RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43757.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43757.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44031.bound LeftAuthority43757.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44031.bound, LeftAuthority43757.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44031.actual selector witness) * (LeftAuthority43757.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44038

namespace LeftBound44039
def owner : Owner := ⟨.program ⟨214⟩, ⟨26592⟩⟩
def transferEvent : Nat := 44039
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩ [⟨.result 43758 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43758 .coefficient)
      LeftAuthority43757.bound (LeftAuthority43757.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26590⟩⟩) (rawTerms := some (Proof.Events170.exact43758RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43757.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43757.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority43757.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43757.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43757.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44039

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
