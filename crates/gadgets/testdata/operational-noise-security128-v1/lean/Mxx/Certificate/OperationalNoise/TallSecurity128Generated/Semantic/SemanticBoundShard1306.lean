import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1305

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound195007
def owner : Owner := ⟨.program ⟨257⟩, ⟨37163⟩⟩
def transferEvent : Nat := 195007
def frameStart : Nat := 194978
def rule : BoundRule := .product (.predecessor 0 195005 .coefficient) (.predecessor 1 195006 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195005 .coefficient)
      LeftAuthority195003.bound (LeftAuthority195003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority195003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority195003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195006 .coefficient)
      LeftAuthority195000.bound (LeftAuthority195000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority195000.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority195000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority195003.bound LeftAuthority195000.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority195003.bound, LeftAuthority195000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority195003.actual selector witness) * (LeftAuthority195000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound195007

namespace LeftBound195011
def owner : Owner := ⟨.program ⟨257⟩, ⟨37164⟩⟩
def transferEvent : Nat := 195011
def frameStart : Nat := 194978
def rule : BoundRule := .identity (.predecessor 0 195010 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195010 .coefficient)
      LeftBound195007.bound (LeftBound195007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195007.derived selector witness)

def rawBound : CoeffClass := LeftBound195007.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound195007.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound195011

namespace LeftBound195028
def owner : Owner := ⟨.program ⟨257⟩, ⟨38714⟩⟩
def transferEvent : Nat := 195028
def frameStart : Nat := 194978
def rule : BoundRule := .sum [.predecessor 0 195026 .coefficient, .predecessor 1 195027 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195026 .coefficient)
      LeftBound195011.bound (LeftBound195011.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound195011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195027 .coefficient)
      LeftAuthority195024.bound (LeftAuthority195024.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority195024.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound195011.bound, LeftAuthority195024.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195011.bound, LeftAuthority195024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound195011.actual selector witness, LeftAuthority195024.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound195028

namespace LeftBound195031
def owner : Owner := ⟨.program ⟨257⟩, ⟨38715⟩⟩
def transferEvent : Nat := 195031
def frameStart : Nat := 194978
def rule : BoundRule := .identity (.predecessor 0 195030 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195030 .coefficient)
      LeftBound195028.bound (LeftBound195028.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound195028.derived selector witness)

def rawBound : CoeffClass := LeftBound195028.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound195028.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound195031

namespace LeftBound195037
def owner : Owner := ⟨.program ⟨257⟩, ⟨38716⟩⟩
def transferEvent : Nat := 195037
def frameStart : Nat := 194978
def rule : BoundRule := .product (.predecessor 0 195035 .coefficient) (.predecessor 1 195036 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195035 .coefficient)
      LeftAuthority195033.bound (LeftAuthority195033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority195033.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority195033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195036 .coefficient)
      LeftBound195031.bound (LeftBound195031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195031.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority195033.bound LeftBound195031.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority195033.bound, LeftBound195031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority195033.actual selector witness) * (LeftBound195031.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound195037

namespace LeftBound195053
def owner : Owner := ⟨.program ⟨257⟩, ⟨9554⟩⟩
def transferEvent : Nat := 195053
def frameStart : Nat := 194978
def rule : BoundRule := .scale (.predecessor 0 195051 .coefficient) (.value (.predecessor 1 195052 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195051 .coefficient)
      LeftAuthority195049.bound (LeftAuthority195049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority195049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority195049.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195052 .coefficient)
      LeftAuthority195040.bound (LeftAuthority195040.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority195040.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority195049.bound LeftAuthority195040.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority195049.bound, LeftAuthority195040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority195049.actual selector witness) * (LeftAuthority195040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound195053

namespace LeftBound195056
def owner : Owner := ⟨.program ⟨257⟩, ⟨7298⟩⟩
def transferEvent : Nat := 195056
def frameStart : Nat := 194978
def rule : BoundRule := .identity (.predecessor 0 195055 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195055 .coefficient)
      LeftAuthority195043.bound (LeftAuthority195043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority195043.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority195043.derived selector witness)

def rawBound : CoeffClass := LeftAuthority195043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority195043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority195043.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound195056

namespace LeftBound195060
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def transferEvent : Nat := 195060
def frameStart : Nat := 194978
def rule : BoundRule := .product (.predecessor 0 195058 .coefficient) (.predecessor 1 195059 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195058 .coefficient)
      LeftBound195056.bound (LeftBound195056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195059 .coefficient)
      LeftBound195053.bound (LeftBound195053.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195053.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195053.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound195056.bound LeftBound195053.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195056.bound, LeftBound195053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound195056.actual selector witness) * (LeftBound195053.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound195060

namespace LeftBound195065
def owner : Owner := ⟨.program ⟨257⟩, ⟨38717⟩⟩
def transferEvent : Nat := 195065
def frameStart : Nat := 194978
def rule : BoundRule := .sum [.predecessor 0 195063 .coefficient, .predecessor 1 195064 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195063 .coefficient)
      LeftBound195060.bound (LeftBound195060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195062RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195060.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195060.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195064 .coefficient)
      LeftBound195037.bound (LeftBound195037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195037.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195037.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound195060.bound, LeftBound195037.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195060.bound, LeftBound195037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound195060.actual selector witness, LeftBound195037.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound195065

namespace LeftBound195069
def owner : Owner := ⟨.program ⟨257⟩, ⟨38964⟩⟩
def transferEvent : Nat := 195069
def frameStart : Nat := 194978
def rule : BoundRule := .product (.predecessor 0 195067 .coefficient) (.predecessor 1 195068 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195067 .coefficient)
      LeftBound195065.bound (LeftBound195065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195065.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195065.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195068 .coefficient)
      LeftAuthority195022.bound (LeftAuthority195022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority195022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority195022.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound195065.bound LeftAuthority195022.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195065.bound, LeftAuthority195022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound195065.actual selector witness) * (LeftAuthority195022.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound195069

namespace LeftBound195080
def owner : Owner := ⟨.program ⟨257⟩, ⟨37446⟩⟩
def transferEvent : Nat := 195080
def frameStart : Nat := 194978
def rule : BoundRule := .product (.predecessor 0 195078 .coefficient) (.predecessor 1 195079 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195078 .coefficient)
      LeftAuthority195033.bound (LeftAuthority195033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact195034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority195033.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority195033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195079 .coefficient)
      LeftAuthority195076.bound (LeftAuthority195076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events762.exact195077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority195076.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority195076.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority195033.bound LeftAuthority195076.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority195033.bound, LeftAuthority195076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority195033.actual selector witness) * (LeftAuthority195076.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound195080

namespace LeftBound195088
def owner : Owner := ⟨.program ⟨257⟩, ⟨37447⟩⟩
def transferEvent : Nat := 195088
def frameStart : Nat := 194978
def rule : BoundRule := .sum [.predecessor 0 195086 .coefficient, .predecessor 1 195087 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195086 .coefficient)
      LeftAuthority195084.bound (LeftAuthority195084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events762.exact195085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority195084.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority195084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195087 .coefficient)
      LeftBound195080.bound (LeftBound195080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events762.exact195082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195080.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195080.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority195084.bound, LeftBound195080.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority195084.bound, LeftBound195080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority195084.actual selector witness, LeftBound195080.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound195088

namespace LeftBound195092
def owner : Owner := ⟨.program ⟨257⟩, ⟨38965⟩⟩
def transferEvent : Nat := 195092
def frameStart : Nat := 194978
def rule : BoundRule := .sum [.predecessor 0 195090 .coefficient, .predecessor 1 195091 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195090 .coefficient)
      LeftBound195088.bound (LeftBound195088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events762.exact195089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195088.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195088.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195091 .coefficient)
      LeftBound195069.bound (LeftBound195069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events762.exact195074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195069.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195069.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound195088.bound, LeftBound195069.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195088.bound, LeftBound195069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound195088.actual selector witness, LeftBound195069.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound195092

namespace LeftBound195105
def owner : Owner := ⟨.program ⟨257⟩, ⟨38963⟩⟩
def transferEvent : Nat := 195105
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 195103 .coefficient, .predecessor 1 195104 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195103 .coefficient)
      LeftBound194926.bound (LeftBound194926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events762.exact195102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195104 .coefficient)
      LeftBound194909.bound (LeftBound194909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact194916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194909.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound194926.bound, LeftBound194909.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194926.bound, LeftBound194909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound194926.actual selector witness, LeftBound194909.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound195105

namespace LeftBound195108
def owner : Owner := ⟨.program ⟨257⟩, ⟨38963⟩⟩
def transferEvent : Nat := 195108
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 195102 .summary, .result 194916 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 195102 .summary)
      LeftBound194928.bound (LeftBound194928.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37892⟩⟩) (rawTerms := some (Proof.Events762.exact195102RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound194928.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 194916 .summary)
      LeftBound194911.bound (LeftBound194911.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38962⟩⟩) (rawTerms := some (Proof.Events761.exact194916RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound194911.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound194928.bound, LeftBound194911.bound]
def bound : CoeffClass := .finite ⟨2998182198162866044928, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194928.bound, LeftBound194911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound194928.actual selector witness, LeftBound194911.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound195108

namespace LeftBound195112
def owner : Owner := ⟨.program ⟨257⟩, ⟨39361⟩⟩
def transferEvent : Nat := 195112
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 195110 .coefficient) (.predecessor 1 195111 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 195110 .coefficient)
      LeftBound195105.bound (LeftBound195105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events762.exact195109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 195111 .coefficient)
      LeftAuthority194831.bound (LeftAuthority194831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact194832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority194831.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority194831.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound195105.bound LeftAuthority194831.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195105.bound, LeftAuthority194831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound195105.actual selector witness) * (LeftAuthority194831.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound195112

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
