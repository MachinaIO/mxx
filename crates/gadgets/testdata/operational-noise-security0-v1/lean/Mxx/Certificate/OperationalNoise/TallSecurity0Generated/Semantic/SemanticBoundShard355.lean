import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard354

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound52795
def owner : Owner := ⟨.program ⟨214⟩, ⟨12666⟩⟩
def transferEvent : Nat := 52795
def frameStart : Nat := 52745
def rule : BoundRule := .sum [.predecessor 0 52793 .coefficient, .predecessor 1 52794 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52793 .coefficient)
      LeftBound52778.bound (LeftBound52778.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52778.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52794 .coefficient)
      LeftAuthority52791.bound (LeftAuthority52791.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority52791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52778.bound, LeftAuthority52791.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52778.bound, LeftAuthority52791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52778.actual selector witness, LeftAuthority52791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52795

namespace LeftBound52798
def owner : Owner := ⟨.program ⟨214⟩, ⟨12667⟩⟩
def transferEvent : Nat := 52798
def frameStart : Nat := 52745
def rule : BoundRule := .identity (.predecessor 0 52797 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52797 .coefficient)
      LeftBound52795.bound (LeftBound52795.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52795.derived selector witness)

def rawBound : CoeffClass := LeftBound52795.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound52795.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52798

namespace LeftBound52804
def owner : Owner := ⟨.program ⟨214⟩, ⟨12668⟩⟩
def transferEvent : Nat := 52804
def frameStart : Nat := 52745
def rule : BoundRule := .product (.predecessor 0 52802 .coefficient) (.predecessor 1 52803 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52802 .coefficient)
      LeftAuthority52800.bound (LeftAuthority52800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52800.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52803 .coefficient)
      LeftBound52798.bound (LeftBound52798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority52800.bound LeftBound52798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52800.bound, LeftBound52798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority52800.actual selector witness) * (LeftBound52798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52804

namespace LeftBound52820
def owner : Owner := ⟨.program ⟨214⟩, ⟨7871⟩⟩
def transferEvent : Nat := 52820
def frameStart : Nat := 52745
def rule : BoundRule := .scale (.predecessor 0 52818 .coefficient) (.value (.predecessor 1 52819 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52818 .coefficient)
      LeftAuthority52816.bound (LeftAuthority52816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52819 .coefficient)
      LeftAuthority52807.bound (LeftAuthority52807.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority52807.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority52816.bound LeftAuthority52807.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52816.bound, LeftAuthority52807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52816.actual selector witness) * (LeftAuthority52807.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound52820

namespace LeftBound52823
def owner : Owner := ⟨.program ⟨214⟩, ⟨6766⟩⟩
def transferEvent : Nat := 52823
def frameStart : Nat := 52745
def rule : BoundRule := .identity (.predecessor 0 52822 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52822 .coefficient)
      LeftAuthority52810.bound (LeftAuthority52810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52810.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52810.derived selector witness)

def rawBound : CoeffClass := LeftAuthority52810.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority52810.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52823

namespace LeftBound52827
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def transferEvent : Nat := 52827
def frameStart : Nat := 52745
def rule : BoundRule := .product (.predecessor 0 52825 .coefficient) (.predecessor 1 52826 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52825 .coefficient)
      LeftBound52823.bound (LeftBound52823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52826 .coefficient)
      LeftBound52820.bound (LeftBound52820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52820.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52823.bound LeftBound52820.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52823.bound, LeftBound52820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52823.actual selector witness) * (LeftBound52820.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52827

namespace LeftBound52832
def owner : Owner := ⟨.program ⟨214⟩, ⟨12669⟩⟩
def transferEvent : Nat := 52832
def frameStart : Nat := 52745
def rule : BoundRule := .sum [.predecessor 0 52830 .coefficient, .predecessor 1 52831 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52830 .coefficient)
      LeftBound52827.bound (LeftBound52827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52827.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52831 .coefficient)
      LeftBound52804.bound (LeftBound52804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52804.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52804.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52827.bound, LeftBound52804.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52827.bound, LeftBound52804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52827.actual selector witness, LeftBound52804.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52832

namespace LeftBound52836
def owner : Owner := ⟨.program ⟨214⟩, ⟨25458⟩⟩
def transferEvent : Nat := 52836
def frameStart : Nat := 52745
def rule : BoundRule := .product (.predecessor 0 52834 .coefficient) (.predecessor 1 52835 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52834 .coefficient)
      LeftBound52832.bound (LeftBound52832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52832.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52835 .coefficient)
      LeftAuthority52789.bound (LeftAuthority52789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52789.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52789.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52832.bound LeftAuthority52789.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52832.bound, LeftAuthority52789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52832.actual selector witness) * (LeftAuthority52789.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52836

namespace LeftBound52847
def owner : Owner := ⟨.program ⟨214⟩, ⟨16555⟩⟩
def transferEvent : Nat := 52847
def frameStart : Nat := 52745
def rule : BoundRule := .product (.predecessor 0 52845 .coefficient) (.predecessor 1 52846 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52845 .coefficient)
      LeftAuthority52800.bound (LeftAuthority52800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52800.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52846 .coefficient)
      LeftAuthority52843.bound (LeftAuthority52843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52843.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority52800.bound LeftAuthority52843.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52800.bound, LeftAuthority52843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority52800.actual selector witness) * (LeftAuthority52843.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52847

namespace LeftBound52855
def owner : Owner := ⟨.program ⟨214⟩, ⟨16556⟩⟩
def transferEvent : Nat := 52855
def frameStart : Nat := 52745
def rule : BoundRule := .sum [.predecessor 0 52853 .coefficient, .predecessor 1 52854 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52853 .coefficient)
      LeftAuthority52851.bound (LeftAuthority52851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52854 .coefficient)
      LeftBound52847.bound (LeftBound52847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52847.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52847.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority52851.bound, LeftBound52847.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52851.bound, LeftBound52847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority52851.actual selector witness, LeftBound52847.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52855

namespace LeftBound52859
def owner : Owner := ⟨.program ⟨214⟩, ⟨25459⟩⟩
def transferEvent : Nat := 52859
def frameStart : Nat := 52745
def rule : BoundRule := .sum [.predecessor 0 52857 .coefficient, .predecessor 1 52858 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52857 .coefficient)
      LeftBound52855.bound (LeftBound52855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52858 .coefficient)
      LeftBound52836.bound (LeftBound52836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52855.bound, LeftBound52836.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52855.bound, LeftBound52836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52855.actual selector witness, LeftBound52836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52859

namespace LeftBound52872
def owner : Owner := ⟨.program ⟨214⟩, ⟨25457⟩⟩
def transferEvent : Nat := 52872
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52870 .coefficient, .predecessor 1 52871 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52870 .coefficient)
      LeftBound52693.bound (LeftBound52693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52871 .coefficient)
      LeftBound52676.bound (LeftBound52676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52693.bound, LeftBound52676.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52693.bound, LeftBound52676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52693.actual selector witness, LeftBound52676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52872

namespace LeftBound52875
def owner : Owner := ⟨.program ⟨214⟩, ⟨25457⟩⟩
def transferEvent : Nat := 52875
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 52869 .summary, .result 52683 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52869 .summary)
      LeftBound52695.bound (LeftBound52695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19967⟩⟩) (rawTerms := some (Proof.Events206.exact52869RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52683 .summary)
      LeftBound52678.bound (LeftBound52678.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25456⟩⟩) (rawTerms := some (Proof.Events205.exact52683RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52695.bound, LeftBound52678.bound]
def bound : CoeffClass := .finite ⟨352134001995776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52695.bound, LeftBound52678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52695.actual selector witness, LeftBound52678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52875

namespace LeftBound52879
def owner : Owner := ⟨.program ⟨214⟩, ⟨29183⟩⟩
def transferEvent : Nat := 52879
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52877 .coefficient) (.predecessor 1 52878 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52877 .coefficient)
      LeftBound52872.bound (LeftBound52872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52878 .coefficient)
      LeftAuthority52598.bound (LeftAuthority52598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52598.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52598.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52872.bound LeftAuthority52598.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52872.bound, LeftAuthority52598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52872.actual selector witness) * (LeftAuthority52598.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52879

namespace LeftBound52880
def owner : Owner := ⟨.program ⟨214⟩, ⟨29183⟩⟩
def transferEvent : Nat := 52880
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29181⟩⟩]⟩ [⟨.result 52599 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52599 .coefficient)
      LeftAuthority52598.bound (LeftAuthority52598.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29181⟩⟩) (rawTerms := some (Proof.Events205.exact52599RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52598.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52598.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority52598.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority52598.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound52880

namespace LeftBound52881
def owner : Owner := ⟨.program ⟨214⟩, ⟨29183⟩⟩
def transferEvent : Nat := 52881
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 52876 .summary) (.transfer 52880) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52876 .summary)
      LeftBound52875.bound (LeftBound52875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25457⟩⟩) (rawTerms := some (Proof.Events206.exact52876RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 52880)
      LeftBound52880.bound (LeftBound52880.actual selector witness) := by
  exact .transfer (LeftBound52880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52875.bound LeftBound52880.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52875.bound, LeftBound52880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52875.actual selector witness) * (LeftBound52880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52881

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
