import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard564

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound82976
def owner : Owner := ⟨.program ⟨214⟩, ⟨11958⟩⟩
def transferEvent : Nat := 82976
def frameStart : Nat := 82947
def rule : BoundRule := .product (.predecessor 0 82974 .coefficient) (.predecessor 1 82975 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82974 .coefficient)
      LeftAuthority82972.bound (LeftAuthority82972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact82973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82972.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82972.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82975 .coefficient)
      LeftAuthority82969.bound (LeftAuthority82969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact82970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82969.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82969.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority82972.bound LeftAuthority82969.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82972.bound, LeftAuthority82969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority82972.actual selector witness) * (LeftAuthority82969.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82976

namespace LeftBound82980
def owner : Owner := ⟨.program ⟨214⟩, ⟨11959⟩⟩
def transferEvent : Nat := 82980
def frameStart : Nat := 82947
def rule : BoundRule := .identity (.predecessor 0 82979 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82979 .coefficient)
      LeftBound82976.bound (LeftBound82976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact82978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82976.derived selector witness)

def rawBound : CoeffClass := LeftBound82976.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound82976.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82980

namespace LeftBound82997
def owner : Owner := ⟨.program ⟨214⟩, ⟨12053⟩⟩
def transferEvent : Nat := 82997
def frameStart : Nat := 82947
def rule : BoundRule := .sum [.predecessor 0 82995 .coefficient, .predecessor 1 82996 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82995 .coefficient)
      LeftBound82980.bound (LeftBound82980.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82996 .coefficient)
      LeftAuthority82993.bound (LeftAuthority82993.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority82993.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82980.bound, LeftAuthority82993.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82980.bound, LeftAuthority82993.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82980.actual selector witness, LeftAuthority82993.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82997

namespace LeftBound83000
def owner : Owner := ⟨.program ⟨214⟩, ⟨12054⟩⟩
def transferEvent : Nat := 83000
def frameStart : Nat := 82947
def rule : BoundRule := .identity (.predecessor 0 82999 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82999 .coefficient)
      LeftBound82997.bound (LeftBound82997.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82997.derived selector witness)

def rawBound : CoeffClass := LeftBound82997.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82997.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound82997.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83000

namespace LeftBound83006
def owner : Owner := ⟨.program ⟨214⟩, ⟨12055⟩⟩
def transferEvent : Nat := 83006
def frameStart : Nat := 82947
def rule : BoundRule := .product (.predecessor 0 83004 .coefficient) (.predecessor 1 83005 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83004 .coefficient)
      LeftAuthority83002.bound (LeftAuthority83002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83005 .coefficient)
      LeftBound83000.bound (LeftBound83000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority83002.bound LeftBound83000.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83002.bound, LeftBound83000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority83002.actual selector witness) * (LeftBound83000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83006

namespace LeftBound83020
def owner : Owner := ⟨.program ⟨214⟩, ⟨7865⟩⟩
def transferEvent : Nat := 83020
def frameStart : Nat := 82947
def rule : BoundRule := .scale (.predecessor 0 83018 .coefficient) (.value (.predecessor 1 83019 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83018 .coefficient)
      LeftAuthority83016.bound (LeftAuthority83016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83016.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83019 .coefficient)
      LeftAuthority82950.bound (LeftAuthority82950.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority82950.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority83016.bound LeftAuthority82950.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83016.bound, LeftAuthority82950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83016.actual selector witness) * (LeftAuthority82950.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83020

namespace LeftBound83023
def owner : Owner := ⟨.program ⟨214⟩, ⟨6764⟩⟩
def transferEvent : Nat := 83023
def frameStart : Nat := 82947
def rule : BoundRule := .identity (.predecessor 0 83022 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83022 .coefficient)
      LeftAuthority83010.bound (LeftAuthority83010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83010.derived selector witness)

def rawBound : CoeffClass := LeftAuthority83010.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority83010.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83023

namespace LeftBound83027
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def transferEvent : Nat := 83027
def frameStart : Nat := 82947
def rule : BoundRule := .product (.predecessor 0 83025 .coefficient) (.predecessor 1 83026 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83025 .coefficient)
      LeftBound83023.bound (LeftBound83023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83026 .coefficient)
      LeftBound83020.bound (LeftBound83020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83020.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83023.bound LeftBound83020.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83023.bound, LeftBound83020.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83023.actual selector witness) * (LeftBound83020.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83027

namespace LeftBound83032
def owner : Owner := ⟨.program ⟨214⟩, ⟨12056⟩⟩
def transferEvent : Nat := 83032
def frameStart : Nat := 82947
def rule : BoundRule := .sum [.predecessor 0 83030 .coefficient, .predecessor 1 83031 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83030 .coefficient)
      LeftBound83027.bound (LeftBound83027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83031 .coefficient)
      LeftBound83006.bound (LeftBound83006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83027.bound, LeftBound83006.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83027.bound, LeftBound83006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83027.actual selector witness, LeftBound83006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83032

namespace LeftBound83036
def owner : Owner := ⟨.program ⟨214⟩, ⟨25222⟩⟩
def transferEvent : Nat := 83036
def frameStart : Nat := 82947
def rule : BoundRule := .product (.predecessor 0 83034 .coefficient) (.predecessor 1 83035 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83034 .coefficient)
      LeftBound83032.bound (LeftBound83032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83035 .coefficient)
      LeftAuthority82991.bound (LeftAuthority82991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact82992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82991.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82991.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83032.bound LeftAuthority82991.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83032.bound, LeftAuthority82991.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83032.actual selector witness) * (LeftAuthority82991.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83036

namespace LeftBound83047
def owner : Owner := ⟨.program ⟨214⟩, ⟨16383⟩⟩
def transferEvent : Nat := 83047
def frameStart : Nat := 82947
def rule : BoundRule := .product (.predecessor 0 83045 .coefficient) (.predecessor 1 83046 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83045 .coefficient)
      LeftAuthority83002.bound (LeftAuthority83002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83046 .coefficient)
      LeftAuthority83043.bound (LeftAuthority83043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83043.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83043.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83002.bound LeftAuthority83043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83002.bound, LeftAuthority83043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority83002.actual selector witness) * (LeftAuthority83043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83047

namespace LeftBound83055
def owner : Owner := ⟨.program ⟨214⟩, ⟨16384⟩⟩
def transferEvent : Nat := 83055
def frameStart : Nat := 82947
def rule : BoundRule := .sum [.predecessor 0 83053 .coefficient, .predecessor 1 83054 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83053 .coefficient)
      LeftAuthority83051.bound (LeftAuthority83051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83051.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83054 .coefficient)
      LeftBound83047.bound (LeftBound83047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83047.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority83051.bound, LeftBound83047.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83051.bound, LeftBound83047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority83051.actual selector witness, LeftBound83047.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83055

namespace LeftBound83059
def owner : Owner := ⟨.program ⟨214⟩, ⟨25223⟩⟩
def transferEvent : Nat := 83059
def frameStart : Nat := 82947
def rule : BoundRule := .sum [.predecessor 0 83057 .coefficient, .predecessor 1 83058 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83057 .coefficient)
      LeftBound83055.bound (LeftBound83055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83058 .coefficient)
      LeftBound83036.bound (LeftBound83036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83036.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83055.bound, LeftBound83036.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83055.bound, LeftBound83036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83055.actual selector witness, LeftBound83036.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83059

namespace LeftBound83072
def owner : Owner := ⟨.program ⟨214⟩, ⟨25221⟩⟩
def transferEvent : Nat := 83072
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83070 .coefficient, .predecessor 1 83071 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83070 .coefficient)
      LeftBound82895.bound (LeftBound82895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83071 .coefficient)
      LeftBound82878.bound (LeftBound82878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82878.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82895.bound, LeftBound82878.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82895.bound, LeftBound82878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82895.actual selector witness, LeftBound82878.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83072

namespace LeftBound83075
def owner : Owner := ⟨.program ⟨214⟩, ⟨25221⟩⟩
def transferEvent : Nat := 83075
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 83069 .summary, .result 82885 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83069 .summary)
      LeftBound82897.bound (LeftBound82897.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19819⟩⟩) (rawTerms := some (Proof.Events324.exact83069RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82885 .summary)
      LeftBound82880.bound (LeftBound82880.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25220⟩⟩) (rawTerms := some (Proof.Events323.exact82885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82897.bound, LeftBound82880.bound]
def bound : CoeffClass := .finite ⟨352115681275904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82897.bound, LeftBound82880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82897.actual selector witness, LeftBound82880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83075

namespace LeftBound83079
def owner : Owner := ⟨.program ⟨214⟩, ⟨28736⟩⟩
def transferEvent : Nat := 83079
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83077 .coefficient) (.predecessor 1 83078 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83077 .coefficient)
      LeftBound83072.bound (LeftBound83072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83078 .coefficient)
      LeftAuthority82800.bound (LeftAuthority82800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82800.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82800.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83072.bound LeftAuthority82800.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83072.bound, LeftAuthority82800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83072.actual selector witness) * (LeftAuthority82800.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83079

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
