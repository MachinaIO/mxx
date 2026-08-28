import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1758

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound259794
def owner : Owner := ⟨.program ⟨257⟩, ⟨17106⟩⟩
def transferEvent : Nat := 259794
def frameStart : Nat := 259744
def rule : BoundRule := .sum [.predecessor 0 259792 .coefficient, .predecessor 1 259793 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259792 .coefficient)
      LeftBound259777.bound (LeftBound259777.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound259777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259793 .coefficient)
      LeftAuthority259790.bound (LeftAuthority259790.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority259790.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound259777.bound, LeftAuthority259790.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259777.bound, LeftAuthority259790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound259777.actual selector witness, LeftAuthority259790.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259794

namespace LeftBound259797
def owner : Owner := ⟨.program ⟨257⟩, ⟨17107⟩⟩
def transferEvent : Nat := 259797
def frameStart : Nat := 259744
def rule : BoundRule := .identity (.predecessor 0 259796 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259796 .coefficient)
      LeftBound259794.bound (LeftBound259794.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound259794.derived selector witness)

def rawBound : CoeffClass := LeftBound259794.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound259794.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound259797

namespace LeftBound259803
def owner : Owner := ⟨.program ⟨257⟩, ⟨17108⟩⟩
def transferEvent : Nat := 259803
def frameStart : Nat := 259744
def rule : BoundRule := .product (.predecessor 0 259801 .coefficient) (.predecessor 1 259802 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259801 .coefficient)
      LeftAuthority259799.bound (LeftAuthority259799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259802 .coefficient)
      LeftBound259797.bound (LeftBound259797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259797.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority259799.bound LeftBound259797.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority259799.bound, LeftBound259797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority259799.actual selector witness) * (LeftBound259797.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259803

namespace LeftBound259819
def owner : Owner := ⟨.program ⟨257⟩, ⟨9569⟩⟩
def transferEvent : Nat := 259819
def frameStart : Nat := 259744
def rule : BoundRule := .scale (.predecessor 0 259817 .coefficient) (.value (.predecessor 1 259818 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259817 .coefficient)
      LeftAuthority259815.bound (LeftAuthority259815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259818 .coefficient)
      LeftAuthority259806.bound (LeftAuthority259806.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority259806.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority259815.bound LeftAuthority259806.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority259815.bound, LeftAuthority259806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority259815.actual selector witness) * (LeftAuthority259806.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound259819

namespace LeftBound259822
def owner : Owner := ⟨.program ⟨257⟩, ⟨7303⟩⟩
def transferEvent : Nat := 259822
def frameStart : Nat := 259744
def rule : BoundRule := .identity (.predecessor 0 259821 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259821 .coefficient)
      LeftAuthority259809.bound (LeftAuthority259809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259809.derived selector witness)

def rawBound : CoeffClass := LeftAuthority259809.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority259809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority259809.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound259822

namespace LeftBound259826
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def transferEvent : Nat := 259826
def frameStart : Nat := 259744
def rule : BoundRule := .product (.predecessor 0 259824 .coefficient) (.predecessor 1 259825 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259824 .coefficient)
      LeftBound259822.bound (LeftBound259822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259822.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259825 .coefficient)
      LeftBound259819.bound (LeftBound259819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259819.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259822.bound LeftBound259819.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259822.bound, LeftBound259819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259822.actual selector witness) * (LeftBound259819.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259826

namespace LeftBound259831
def owner : Owner := ⟨.program ⟨257⟩, ⟨17109⟩⟩
def transferEvent : Nat := 259831
def frameStart : Nat := 259744
def rule : BoundRule := .sum [.predecessor 0 259829 .coefficient, .predecessor 1 259830 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259829 .coefficient)
      LeftBound259826.bound (LeftBound259826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259830 .coefficient)
      LeftBound259803.bound (LeftBound259803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259803.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound259826.bound, LeftBound259803.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259826.bound, LeftBound259803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound259826.actual selector witness, LeftBound259803.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259831

namespace LeftBound259835
def owner : Owner := ⟨.program ⟨257⟩, ⟨17307⟩⟩
def transferEvent : Nat := 259835
def frameStart : Nat := 259744
def rule : BoundRule := .product (.predecessor 0 259833 .coefficient) (.predecessor 1 259834 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259833 .coefficient)
      LeftBound259831.bound (LeftBound259831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259831.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259834 .coefficient)
      LeftAuthority259788.bound (LeftAuthority259788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259788.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259788.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259831.bound LeftAuthority259788.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259831.bound, LeftAuthority259788.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259831.actual selector witness) * (LeftAuthority259788.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259835

namespace LeftBound259846
def owner : Owner := ⟨.program ⟨257⟩, ⟨15750⟩⟩
def transferEvent : Nat := 259846
def frameStart : Nat := 259744
def rule : BoundRule := .product (.predecessor 0 259844 .coefficient) (.predecessor 1 259845 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259844 .coefficient)
      LeftAuthority259799.bound (LeftAuthority259799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259845 .coefficient)
      LeftAuthority259842.bound (LeftAuthority259842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact259843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259842.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority259799.bound LeftAuthority259842.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority259799.bound, LeftAuthority259842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority259799.actual selector witness) * (LeftAuthority259842.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259846

namespace LeftBound259854
def owner : Owner := ⟨.program ⟨257⟩, ⟨15751⟩⟩
def transferEvent : Nat := 259854
def frameStart : Nat := 259744
def rule : BoundRule := .sum [.predecessor 0 259852 .coefficient, .predecessor 1 259853 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259852 .coefficient)
      LeftAuthority259850.bound (LeftAuthority259850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact259851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259850.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259853 .coefficient)
      LeftBound259846.bound (LeftBound259846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact259848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259846.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority259850.bound, LeftBound259846.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority259850.bound, LeftBound259846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority259850.actual selector witness, LeftBound259846.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259854

namespace LeftBound259858
def owner : Owner := ⟨.program ⟨257⟩, ⟨17308⟩⟩
def transferEvent : Nat := 259858
def frameStart : Nat := 259744
def rule : BoundRule := .sum [.predecessor 0 259856 .coefficient, .predecessor 1 259857 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259856 .coefficient)
      LeftBound259854.bound (LeftBound259854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact259855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259854.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259857 .coefficient)
      LeftBound259835.bound (LeftBound259835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact259840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound259854.bound, LeftBound259835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259854.bound, LeftBound259835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound259854.actual selector witness, LeftBound259835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259858

namespace LeftBound259871
def owner : Owner := ⟨.program ⟨257⟩, ⟨17306⟩⟩
def transferEvent : Nat := 259871
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 259869 .coefficient, .predecessor 1 259870 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259869 .coefficient)
      LeftBound259692.bound (LeftBound259692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact259868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259870 .coefficient)
      LeftBound259675.bound (LeftBound259675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound259692.bound, LeftBound259675.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259692.bound, LeftBound259675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound259692.actual selector witness, LeftBound259675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259871

namespace LeftBound259874
def owner : Owner := ⟨.program ⟨257⟩, ⟨17306⟩⟩
def transferEvent : Nat := 259874
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 259868 .summary, .result 259682 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259868 .summary)
      LeftBound259694.bound (LeftBound259694.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16242⟩⟩) (rawTerms := some (Proof.Events1015.exact259868RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259682 .summary)
      LeftBound259677.bound (LeftBound259677.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17305⟩⟩) (rawTerms := some (Proof.Events1014.exact259682RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259677.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound259694.bound, LeftBound259677.bound]
def bound : CoeffClass := .finite ⟨2997816280693142192128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259694.bound, LeftBound259677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound259694.actual selector witness, LeftBound259677.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259874

namespace LeftBound259878
def owner : Owner := ⟨.program ⟨257⟩, ⟨17623⟩⟩
def transferEvent : Nat := 259878
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 259876 .coefficient) (.predecessor 1 259877 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259876 .coefficient)
      LeftBound259871.bound (LeftBound259871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact259875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259877 .coefficient)
      LeftAuthority259597.bound (LeftAuthority259597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1014.exact259598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259597.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259871.bound LeftAuthority259597.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259871.bound, LeftAuthority259597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259871.actual selector witness) * (LeftAuthority259597.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259878

namespace LeftBound259879
def owner : Owner := ⟨.program ⟨257⟩, ⟨17623⟩⟩
def transferEvent : Nat := 259879
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩ [⟨.result 259598 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259598 .coefficient)
      LeftAuthority259597.bound (LeftAuthority259597.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨17621⟩⟩) (rawTerms := some (Proof.Events1014.exact259598RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259597.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority259597.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority259597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority259597.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound259879

namespace LeftBound259880
def owner : Owner := ⟨.program ⟨257⟩, ⟨17623⟩⟩
def transferEvent : Nat := 259880
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 259875 .summary) (.transfer 259879) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259875 .summary)
      LeftBound259874.bound (LeftBound259874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17306⟩⟩) (rawTerms := some (Proof.Events1015.exact259875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 259879)
      LeftBound259879.bound (LeftBound259879.actual selector witness) := by
  exact .transfer (LeftBound259879.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259874.bound LeftBound259879.bound
def bound : CoeffClass := .finite ⟨32188807212483504816668771614720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259874.bound, LeftBound259879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259874.actual selector witness) * (LeftBound259879.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259880

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
