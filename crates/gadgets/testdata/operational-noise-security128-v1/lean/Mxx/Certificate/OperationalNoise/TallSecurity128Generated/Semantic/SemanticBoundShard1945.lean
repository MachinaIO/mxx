import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard118
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1895
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1898
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1944

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound286806
def owner : Owner := ⟨.program ⟨257⟩, ⟨53821⟩⟩
def transferEvent : Nat := 286806
def frameStart : Nat := 286767
def rule : BoundRule := .identity (.predecessor 0 286805 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286805 .coefficient)
      LeftAuthority286803.bound (LeftAuthority286803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286803.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286803.derived selector witness)

def rawBound : CoeffClass := LeftAuthority286803.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority286803.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound286806

namespace LeftBound286823
def owner : Owner := ⟨.program ⟨257⟩, ⟨55322⟩⟩
def transferEvent : Nat := 286823
def frameStart : Nat := 286767
def rule : BoundRule := .sum [.predecessor 0 286821 .coefficient, .predecessor 1 286822 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286821 .coefficient)
      LeftBound286806.bound (LeftBound286806.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound286806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286822 .coefficient)
      LeftAuthority286819.bound (LeftAuthority286819.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority286819.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286806.bound, LeftAuthority286819.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286806.bound, LeftAuthority286819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286806.actual selector witness, LeftAuthority286819.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286823

namespace LeftBound286826
def owner : Owner := ⟨.program ⟨257⟩, ⟨55323⟩⟩
def transferEvent : Nat := 286826
def frameStart : Nat := 286767
def rule : BoundRule := .identity (.predecessor 0 286825 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286825 .coefficient)
      LeftBound286823.bound (LeftBound286823.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound286823.derived selector witness)

def rawBound : CoeffClass := LeftBound286823.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286823.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound286823.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound286826

namespace LeftBound286832
def owner : Owner := ⟨.program ⟨257⟩, ⟨55324⟩⟩
def transferEvent : Nat := 286832
def frameStart : Nat := 286767
def rule : BoundRule := .product (.predecessor 0 286830 .coefficient) (.predecessor 1 286831 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286830 .coefficient)
      LeftAuthority286828.bound (LeftAuthority286828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286831 .coefficient)
      LeftBound286826.bound (LeftBound286826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286826.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority286828.bound LeftBound286826.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286828.bound, LeftBound286826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority286828.actual selector witness) * (LeftBound286826.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286832

namespace LeftBound286840
def owner : Owner := ⟨.program ⟨257⟩, ⟨55325⟩⟩
def transferEvent : Nat := 286840
def frameStart : Nat := 286767
def rule : BoundRule := .sum [.predecessor 0 286838 .coefficient, .predecessor 1 286839 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286838 .coefficient)
      LeftAuthority286836.bound (LeftAuthority286836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286839 .coefficient)
      LeftBound286832.bound (LeftBound286832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286832.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286832.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority286836.bound, LeftBound286832.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286836.bound, LeftBound286832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority286836.actual selector witness, LeftBound286832.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286840

namespace LeftBound286844
def owner : Owner := ⟨.program ⟨257⟩, ⟨55747⟩⟩
def transferEvent : Nat := 286844
def frameStart : Nat := 286767
def rule : BoundRule := .product (.predecessor 0 286842 .coefficient) (.predecessor 1 286843 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286842 .coefficient)
      LeftBound286840.bound (LeftBound286840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286840.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286843 .coefficient)
      LeftAuthority286817.bound (LeftAuthority286817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286817.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound286840.bound LeftAuthority286817.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286840.bound, LeftAuthority286817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound286840.actual selector witness) * (LeftAuthority286817.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286844

namespace LeftBound286855
def owner : Owner := ⟨.program ⟨257⟩, ⟨54029⟩⟩
def transferEvent : Nat := 286855
def frameStart : Nat := 286767
def rule : BoundRule := .product (.predecessor 0 286853 .coefficient) (.predecessor 1 286854 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286853 .coefficient)
      LeftAuthority286828.bound (LeftAuthority286828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286854 .coefficient)
      LeftAuthority286851.bound (LeftAuthority286851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286851.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority286828.bound LeftAuthority286851.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286828.bound, LeftAuthority286851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority286828.actual selector witness) * (LeftAuthority286851.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286855

namespace LeftBound286863
def owner : Owner := ⟨.program ⟨257⟩, ⟨54030⟩⟩
def transferEvent : Nat := 286863
def frameStart : Nat := 286767
def rule : BoundRule := .sum [.predecessor 0 286861 .coefficient, .predecessor 1 286862 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286861 .coefficient)
      LeftAuthority286859.bound (LeftAuthority286859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286859.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286862 .coefficient)
      LeftBound286855.bound (LeftBound286855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority286859.bound, LeftBound286855.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286859.bound, LeftBound286855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority286859.actual selector witness, LeftBound286855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286863

namespace LeftBound286867
def owner : Owner := ⟨.program ⟨257⟩, ⟨55751⟩⟩
def transferEvent : Nat := 286867
def frameStart : Nat := 286767
def rule : BoundRule := .sum [.predecessor 0 286865 .coefficient, .predecessor 1 286866 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286865 .coefficient)
      LeftBound286863.bound (LeftBound286863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286866 .coefficient)
      LeftBound286844.bound (LeftBound286844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286844.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286844.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286863.bound, LeftBound286844.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286863.bound, LeftBound286844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286863.actual selector witness, LeftBound286844.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286867

namespace LeftBound286880
def owner : Owner := ⟨.program ⟨257⟩, ⟨55749⟩⟩
def transferEvent : Nat := 286880
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 286878 .coefficient, .predecessor 1 286879 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286878 .coefficient)
      LeftBound286709.bound (LeftBound286709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286879 .coefficient)
      LeftBound286692.bound (LeftBound286692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1119.exact286699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286709.bound, LeftBound286692.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286709.bound, LeftBound286692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286709.actual selector witness, LeftBound286692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286880

namespace LeftBound286883
def owner : Owner := ⟨.program ⟨257⟩, ⟨55749⟩⟩
def transferEvent : Nat := 286883
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 286877 .summary, .result 286699 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286877 .summary)
      LeftBound286711.bound (LeftBound286711.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨54619⟩⟩) (rawTerms := some (Proof.Events1120.exact286877RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286699 .summary)
      LeftBound286694.bound (LeftBound286694.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55748⟩⟩) (rawTerms := some (Proof.Events1119.exact286699RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286694.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286711.bound, LeftBound286694.bound]
def bound : CoeffClass := .finite ⟨32189789464712143775715074244608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286711.bound, LeftBound286694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286711.actual selector witness, LeftBound286694.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286883

namespace LeftBound286907
def owner : Owner := ⟨.program ⟨257⟩, ⟨24459⟩⟩
def transferEvent : Nat := 286907
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 286905 .coefficient) (.predecessor 1 286906 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286905 .coefficient)
      LeftAuthority13850.bound (LeftAuthority13850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13850.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286906 .coefficient)
      LeftBound280651.bound (LeftBound280651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13850.bound LeftBound280651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13850.bound, LeftBound280651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13850.actual selector witness) * (LeftBound280651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound286907

namespace LeftBound286912
def owner : Owner := ⟨.program ⟨257⟩, ⟨7930⟩⟩
def transferEvent : Nat := 286912
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 286910 .coefficient) (.predecessor 1 286911 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286910 .coefficient)
      LeftBound280522.bound (LeftBound280522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1095.exact280523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286911 .coefficient)
      LeftBound23592.bound (LeftBound23592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound280522.bound LeftBound23592.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280522.bound, LeftBound23592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound280522.actual selector witness) * (LeftBound23592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286912

namespace LeftBound286917
def owner : Owner := ⟨.program ⟨257⟩, ⟨24460⟩⟩
def transferEvent : Nat := 286917
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 286915 .coefficient, .predecessor 1 286916 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286915 .coefficient)
      LeftBound286912.bound (LeftBound286912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286916 .coefficient)
      LeftBound286907.bound (LeftBound286907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286912.bound, LeftBound286907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286912.bound, LeftBound286907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286912.actual selector witness, LeftBound286907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286917

namespace LeftBound286921
def owner : Owner := ⟨.program ⟨257⟩, ⟨24461⟩⟩
def transferEvent : Nat := 286921
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 286919 .coefficient, .predecessor 1 286920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286919 .coefficient)
      LeftBound286917.bound (LeftBound286917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286920 .coefficient)
      LeftBound23584.bound (LeftBound23584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23584.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286917.bound, LeftBound23584.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286917.bound, LeftBound23584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286917.actual selector witness, LeftBound23584.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286921

namespace LeftBound286922
def owner : Owner := ⟨.program ⟨257⟩, ⟨24461⟩⟩
def transferEvent : Nat := 286922
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨134⟩⟩]⟩ [⟨.result 23585 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23585 .coefficient)
      LeftBound23584.bound (LeftBound23584.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨134⟩⟩) (rawTerms := some (Proof.Events092.exact23585RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23584.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound23584.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound23584.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound286922

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
