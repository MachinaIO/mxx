import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard659

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound96713
def owner : Owner := ⟨.program ⟨214⟩, ⟨12458⟩⟩
def transferEvent : Nat := 96713
def frameStart : Nat := 96675
def rule : BoundRule := .sum [.predecessor 0 96711 .coefficient, .predecessor 1 96712 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96711 .coefficient)
      LeftBound96696.bound (LeftBound96696.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96712 .coefficient)
      LeftAuthority96709.bound (LeftAuthority96709.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority96709.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96696.bound, LeftAuthority96709.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96696.bound, LeftAuthority96709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96696.actual selector witness, LeftAuthority96709.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96713

namespace LeftBound96716
def owner : Owner := ⟨.program ⟨214⟩, ⟨12459⟩⟩
def transferEvent : Nat := 96716
def frameStart : Nat := 96675
def rule : BoundRule := .identity (.predecessor 0 96715 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96715 .coefficient)
      LeftBound96713.bound (LeftBound96713.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96713.derived selector witness)

def rawBound : CoeffClass := LeftBound96713.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound96713.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96716

namespace LeftBound96722
def owner : Owner := ⟨.program ⟨214⟩, ⟨12460⟩⟩
def transferEvent : Nat := 96722
def frameStart : Nat := 96675
def rule : BoundRule := .product (.predecessor 0 96720 .coefficient) (.predecessor 1 96721 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96720 .coefficient)
      LeftAuthority96718.bound (LeftAuthority96718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96721 .coefficient)
      LeftBound96716.bound (LeftBound96716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96716.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority96718.bound LeftBound96716.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96718.bound, LeftBound96716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority96718.actual selector witness) * (LeftBound96716.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96722

namespace LeftBound96738
def owner : Owner := ⟨.program ⟨214⟩, ⟨7868⟩⟩
def transferEvent : Nat := 96738
def frameStart : Nat := 96675
def rule : BoundRule := .scale (.predecessor 0 96736 .coefficient) (.value (.predecessor 1 96737 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96736 .coefficient)
      LeftAuthority96734.bound (LeftAuthority96734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96737 .coefficient)
      LeftAuthority96725.bound (LeftAuthority96725.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority96725.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority96734.bound LeftAuthority96725.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96734.bound, LeftAuthority96725.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96734.actual selector witness) * (LeftAuthority96725.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96738

namespace LeftBound96741
def owner : Owner := ⟨.program ⟨214⟩, ⟨6765⟩⟩
def transferEvent : Nat := 96741
def frameStart : Nat := 96675
def rule : BoundRule := .identity (.predecessor 0 96740 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96740 .coefficient)
      LeftAuthority96728.bound (LeftAuthority96728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96728.derived selector witness)

def rawBound : CoeffClass := LeftAuthority96728.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority96728.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96741

namespace LeftBound96745
def owner : Owner := ⟨.program ⟨214⟩, ⟨7869⟩⟩
def transferEvent : Nat := 96745
def frameStart : Nat := 96675
def rule : BoundRule := .product (.predecessor 0 96743 .coefficient) (.predecessor 1 96744 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96743 .coefficient)
      LeftBound96741.bound (LeftBound96741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96744 .coefficient)
      LeftBound96738.bound (LeftBound96738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96738.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96741.bound LeftBound96738.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96741.bound, LeftBound96738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96741.actual selector witness) * (LeftBound96738.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96745

namespace LeftBound96750
def owner : Owner := ⟨.program ⟨214⟩, ⟨12461⟩⟩
def transferEvent : Nat := 96750
def frameStart : Nat := 96675
def rule : BoundRule := .sum [.predecessor 0 96748 .coefficient, .predecessor 1 96749 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96748 .coefficient)
      LeftBound96745.bound (LeftBound96745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96745.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96749 .coefficient)
      LeftBound96722.bound (LeftBound96722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96722.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96745.bound, LeftBound96722.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96745.bound, LeftBound96722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96745.actual selector witness, LeftBound96722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96750

namespace LeftBound96754
def owner : Owner := ⟨.program ⟨214⟩, ⟨25363⟩⟩
def transferEvent : Nat := 96754
def frameStart : Nat := 96675
def rule : BoundRule := .product (.predecessor 0 96752 .coefficient) (.predecessor 1 96753 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96752 .coefficient)
      LeftBound96750.bound (LeftBound96750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96753 .coefficient)
      LeftAuthority96707.bound (LeftAuthority96707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96707.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96707.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96750.bound LeftAuthority96707.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96750.bound, LeftAuthority96707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96750.actual selector witness) * (LeftAuthority96707.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96754

namespace LeftBound96765
def owner : Owner := ⟨.program ⟨214⟩, ⟨16457⟩⟩
def transferEvent : Nat := 96765
def frameStart : Nat := 96675
def rule : BoundRule := .product (.predecessor 0 96763 .coefficient) (.predecessor 1 96764 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96763 .coefficient)
      LeftAuthority96718.bound (LeftAuthority96718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96764 .coefficient)
      LeftAuthority96761.bound (LeftAuthority96761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96761.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96761.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority96718.bound LeftAuthority96761.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96718.bound, LeftAuthority96761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority96718.actual selector witness) * (LeftAuthority96761.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96765

namespace LeftBound96773
def owner : Owner := ⟨.program ⟨214⟩, ⟨16458⟩⟩
def transferEvent : Nat := 96773
def frameStart : Nat := 96675
def rule : BoundRule := .sum [.predecessor 0 96771 .coefficient, .predecessor 1 96772 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96771 .coefficient)
      LeftAuthority96769.bound (LeftAuthority96769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96769.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96772 .coefficient)
      LeftBound96765.bound (LeftBound96765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96765.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority96769.bound, LeftBound96765.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96769.bound, LeftBound96765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority96769.actual selector witness, LeftBound96765.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96773

namespace LeftBound96777
def owner : Owner := ⟨.program ⟨214⟩, ⟨25364⟩⟩
def transferEvent : Nat := 96777
def frameStart : Nat := 96675
def rule : BoundRule := .sum [.predecessor 0 96775 .coefficient, .predecessor 1 96776 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96775 .coefficient)
      LeftBound96773.bound (LeftBound96773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96776 .coefficient)
      LeftBound96754.bound (LeftBound96754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96754.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96773.bound, LeftBound96754.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96773.bound, LeftBound96754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96773.actual selector witness, LeftBound96754.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96777

namespace LeftBound96790
def owner : Owner := ⟨.program ⟨214⟩, ⟨25362⟩⟩
def transferEvent : Nat := 96790
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96788 .coefficient, .predecessor 1 96789 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96788 .coefficient)
      LeftBound96635.bound (LeftBound96635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96635.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96789 .coefficient)
      LeftBound96618.bound (LeftBound96618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96618.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96635.bound, LeftBound96618.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96635.bound, LeftBound96618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96635.actual selector witness, LeftBound96618.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96790

namespace LeftBound96793
def owner : Owner := ⟨.program ⟨214⟩, ⟨25362⟩⟩
def transferEvent : Nat := 96793
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 96787 .summary, .result 96625 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96787 .summary)
      LeftBound96637.bound (LeftBound96637.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19880⟩⟩) (rawTerms := some (Proof.Events378.exact96787RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96625 .summary)
      LeftBound96620.bound (LeftBound96620.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25361⟩⟩) (rawTerms := some (Proof.Events377.exact96625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96637.bound, LeftBound96620.bound]
def bound : CoeffClass := .finite ⟨352127895089152, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96637.bound, LeftBound96620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96637.actual selector witness, LeftBound96620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96793

namespace LeftBound96797
def owner : Owner := ⟨.program ⟨214⟩, ⟨28918⟩⟩
def transferEvent : Nat := 96797
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96795 .coefficient) (.predecessor 1 96796 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96795 .coefficient)
      LeftBound96790.bound (LeftBound96790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96790.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96790.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96796 .coefficient)
      LeftAuthority96540.bound (LeftAuthority96540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96540.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96540.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96790.bound LeftAuthority96540.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96790.bound, LeftAuthority96540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96790.actual selector witness) * (LeftAuthority96540.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96797

namespace LeftBound96798
def owner : Owner := ⟨.program ⟨214⟩, ⟨28918⟩⟩
def transferEvent : Nat := 96798
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩ [⟨.result 96541 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96541 .coefficient)
      LeftAuthority96540.bound (LeftAuthority96540.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28916⟩⟩) (rawTerms := some (Proof.Events377.exact96541RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96540.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96540.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96540.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96540.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96798

namespace LeftBound96799
def owner : Owner := ⟨.program ⟨214⟩, ⟨28918⟩⟩
def transferEvent : Nat := 96799
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 96794 .summary) (.transfer 96798) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96794 .summary)
      LeftBound96793.bound (LeftBound96793.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25362⟩⟩) (rawTerms := some (Proof.Events378.exact96794RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96793.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96798)
      LeftBound96798.bound (LeftBound96798.actual selector witness) := by
  exact .transfer (LeftBound96798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96793.bound LeftBound96798.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96793.bound, LeftBound96798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96793.actual selector witness) * (LeftBound96798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96799

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
