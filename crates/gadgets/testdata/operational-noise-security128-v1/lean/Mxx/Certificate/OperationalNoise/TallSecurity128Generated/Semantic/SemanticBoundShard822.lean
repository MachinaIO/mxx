import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard110
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard779
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard782
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard821

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound125010
def owner : Owner := ⟨.program ⟨257⟩, ⟨61290⟩⟩
def transferEvent : Nat := 125010
def frameStart : Nat := 124954
def rule : BoundRule := .sum [.predecessor 0 125008 .coefficient, .predecessor 1 125009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125008 .coefficient)
      LeftBound124993.bound (LeftBound124993.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound124993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125009 .coefficient)
      LeftAuthority125006.bound (LeftAuthority125006.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority125006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound124993.bound, LeftAuthority125006.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound124993.bound, LeftAuthority125006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound124993.actual selector witness, LeftAuthority125006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125010

namespace LeftBound125013
def owner : Owner := ⟨.program ⟨257⟩, ⟨61291⟩⟩
def transferEvent : Nat := 125013
def frameStart : Nat := 124954
def rule : BoundRule := .identity (.predecessor 0 125012 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125012 .coefficient)
      LeftBound125010.bound (LeftBound125010.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound125010.derived selector witness)

def rawBound : CoeffClass := LeftBound125010.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound125010.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound125013

namespace LeftBound125019
def owner : Owner := ⟨.program ⟨257⟩, ⟨61292⟩⟩
def transferEvent : Nat := 125019
def frameStart : Nat := 124954
def rule : BoundRule := .product (.predecessor 0 125017 .coefficient) (.predecessor 1 125018 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125017 .coefficient)
      LeftAuthority125015.bound (LeftAuthority125015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority125015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority125015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125018 .coefficient)
      LeftBound125013.bound (LeftBound125013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority125015.bound LeftBound125013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority125015.bound, LeftBound125013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority125015.actual selector witness) * (LeftBound125013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound125019

namespace LeftBound125027
def owner : Owner := ⟨.program ⟨257⟩, ⟨61293⟩⟩
def transferEvent : Nat := 125027
def frameStart : Nat := 124954
def rule : BoundRule := .sum [.predecessor 0 125025 .coefficient, .predecessor 1 125026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125025 .coefficient)
      LeftAuthority125023.bound (LeftAuthority125023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority125023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority125023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125026 .coefficient)
      LeftBound125019.bound (LeftBound125019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority125023.bound, LeftBound125019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority125023.bound, LeftBound125019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority125023.actual selector witness, LeftBound125019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125027

namespace LeftBound125031
def owner : Owner := ⟨.program ⟨257⟩, ⟨61769⟩⟩
def transferEvent : Nat := 125031
def frameStart : Nat := 124954
def rule : BoundRule := .product (.predecessor 0 125029 .coefficient) (.predecessor 1 125030 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125029 .coefficient)
      LeftBound125027.bound (LeftBound125027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125030 .coefficient)
      LeftAuthority125004.bound (LeftAuthority125004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority125004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority125004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound125027.bound LeftAuthority125004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125027.bound, LeftAuthority125004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound125027.actual selector witness) * (LeftAuthority125004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound125031

namespace LeftBound125042
def owner : Owner := ⟨.program ⟨257⟩, ⟨60027⟩⟩
def transferEvent : Nat := 125042
def frameStart : Nat := 124954
def rule : BoundRule := .product (.predecessor 0 125040 .coefficient) (.predecessor 1 125041 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125040 .coefficient)
      LeftAuthority125015.bound (LeftAuthority125015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority125015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority125015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125041 .coefficient)
      LeftAuthority125038.bound (LeftAuthority125038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority125038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority125038.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority125015.bound LeftAuthority125038.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority125015.bound, LeftAuthority125038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority125015.actual selector witness) * (LeftAuthority125038.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound125042

namespace LeftBound125050
def owner : Owner := ⟨.program ⟨257⟩, ⟨60028⟩⟩
def transferEvent : Nat := 125050
def frameStart : Nat := 124954
def rule : BoundRule := .sum [.predecessor 0 125048 .coefficient, .predecessor 1 125049 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125048 .coefficient)
      LeftAuthority125046.bound (LeftAuthority125046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority125046.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority125046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125049 .coefficient)
      LeftBound125042.bound (LeftBound125042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority125046.bound, LeftBound125042.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority125046.bound, LeftBound125042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority125046.actual selector witness, LeftBound125042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125050

namespace LeftBound125054
def owner : Owner := ⟨.program ⟨257⟩, ⟨61773⟩⟩
def transferEvent : Nat := 125054
def frameStart : Nat := 124954
def rule : BoundRule := .sum [.predecessor 0 125052 .coefficient, .predecessor 1 125053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125052 .coefficient)
      LeftBound125050.bound (LeftBound125050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125053 .coefficient)
      LeftBound125031.bound (LeftBound125031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125050.bound, LeftBound125031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125050.bound, LeftBound125031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125050.actual selector witness, LeftBound125031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125054

namespace LeftBound125067
def owner : Owner := ⟨.program ⟨257⟩, ⟨61771⟩⟩
def transferEvent : Nat := 125067
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 125065 .coefficient, .predecessor 1 125066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125065 .coefficient)
      LeftBound124896.bound (LeftBound124896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound124896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound124896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125066 .coefficient)
      LeftBound124879.bound (LeftBound124879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events487.exact124886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound124879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound124879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound124896.bound, LeftBound124879.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound124896.bound, LeftBound124879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound124896.actual selector witness, LeftBound124879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125067

namespace LeftBound125070
def owner : Owner := ⟨.program ⟨257⟩, ⟨61771⟩⟩
def transferEvent : Nat := 125070
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 125064 .summary, .result 124886 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 125064 .summary)
      LeftBound124898.bound (LeftBound124898.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨60619⟩⟩) (rawTerms := some (Proof.Events488.exact125064RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound124898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 124886 .summary)
      LeftBound124881.bound (LeftBound124881.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61770⟩⟩) (rawTerms := some (Proof.Events487.exact124886RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound124881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound124898.bound, LeftBound124881.bound]
def bound : CoeffClass := .finite ⟨32190378816049205907437743505408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound124898.bound, LeftBound124881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound124898.actual selector witness, LeftBound124881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125070

namespace LeftBound125094
def owner : Owner := ⟨.program ⟨257⟩, ⟨24963⟩⟩
def transferEvent : Nat := 125094
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 125092 .coefficient) (.predecessor 1 125093 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125092 .coefficient)
      LeftAuthority5582.bound (LeftAuthority5582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125093 .coefficient)
      LeftBound119776.bound (LeftBound119776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority5582.bound LeftBound119776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5582.bound, LeftBound119776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority5582.actual selector witness) * (LeftBound119776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound125094

namespace LeftBound125099
def owner : Owner := ⟨.program ⟨257⟩, ⟨8123⟩⟩
def transferEvent : Nat := 125099
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 125097 .coefficient) (.predecessor 1 125098 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125097 .coefficient)
      LeftBound119647.bound (LeftBound119647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125098 .coefficient)
      LeftBound22590.bound (LeftBound22590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22590.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound119647.bound LeftBound22590.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119647.bound, LeftBound22590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound119647.actual selector witness) * (LeftBound22590.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound125099

namespace LeftBound125104
def owner : Owner := ⟨.program ⟨257⟩, ⟨24964⟩⟩
def transferEvent : Nat := 125104
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 125102 .coefficient, .predecessor 1 125103 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125102 .coefficient)
      LeftBound125099.bound (LeftBound125099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125103 .coefficient)
      LeftBound125094.bound (LeftBound125094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125099.bound, LeftBound125094.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125099.bound, LeftBound125094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125099.actual selector witness, LeftBound125094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125104

namespace LeftBound125108
def owner : Owner := ⟨.program ⟨257⟩, ⟨24965⟩⟩
def transferEvent : Nat := 125108
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 125106 .coefficient, .predecessor 1 125107 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125106 .coefficient)
      LeftBound125104.bound (LeftBound125104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125107 .coefficient)
      LeftBound22582.bound (LeftBound22582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22582.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound125104.bound, LeftBound22582.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125104.bound, LeftBound22582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound125104.actual selector witness, LeftBound22582.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound125108

namespace LeftBound125109
def owner : Owner := ⟨.program ⟨257⟩, ⟨24965⟩⟩
def transferEvent : Nat := 125109
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨99⟩⟩]⟩ [⟨.result 22583 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22583 .coefficient)
      LeftBound22582.bound (LeftBound22582.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨99⟩⟩) (rawTerms := some (Proof.Events088.exact22583RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22582.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound22582.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound22582.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound125109

namespace LeftBound125114
def owner : Owner := ⟨.program ⟨257⟩, ⟨56400⟩⟩
def transferEvent : Nat := 125114
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 125112 .coefficient) (.predecessor 1 125113 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 125112 .coefficient)
      LeftBound125108.bound (LeftBound125108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events488.exact125111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound125108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound125108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 125113 .coefficient)
      LeftAuthority5585.bound (LeftAuthority5585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5585.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound125108.bound LeftAuthority5585.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound125108.bound, LeftAuthority5585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound125108.actual selector witness) * (LeftAuthority5585.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound125114

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
