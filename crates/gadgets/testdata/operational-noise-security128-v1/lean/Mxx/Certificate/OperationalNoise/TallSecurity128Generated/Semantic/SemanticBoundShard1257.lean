import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1256

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound188192
def owner : Owner := ⟨.program ⟨257⟩, ⟨66815⟩⟩
def transferEvent : Nat := 188192
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188190 .coefficient, .predecessor 1 188191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188190 .coefficient)
      LeftBound188188.bound (LeftBound188188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188191 .coefficient)
      LeftAuthority187868.bound (LeftAuthority187868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events733.exact187869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187868.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188188.bound, LeftAuthority187868.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188188.bound, LeftAuthority187868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188188.actual selector witness, LeftAuthority187868.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188192

namespace LeftBound188196
def owner : Owner := ⟨.program ⟨257⟩, ⟨66816⟩⟩
def transferEvent : Nat := 188196
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188194 .coefficient, .predecessor 1 188195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188194 .coefficient)
      LeftBound188192.bound (LeftBound188192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188195 .coefficient)
      LeftAuthority187845.bound (LeftAuthority187845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events733.exact187846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187845.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188192.bound, LeftAuthority187845.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188192.bound, LeftAuthority187845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188192.actual selector witness, LeftAuthority187845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188196

namespace LeftBound188200
def owner : Owner := ⟨.program ⟨257⟩, ⟨66817⟩⟩
def transferEvent : Nat := 188200
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188198 .coefficient, .predecessor 1 188199 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188198 .coefficient)
      LeftBound188196.bound (LeftBound188196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188199 .coefficient)
      LeftAuthority187822.bound (LeftAuthority187822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events733.exact187823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187822.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188196.bound, LeftAuthority187822.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188196.bound, LeftAuthority187822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188196.actual selector witness, LeftAuthority187822.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188200

namespace LeftBound188204
def owner : Owner := ⟨.program ⟨257⟩, ⟨66818⟩⟩
def transferEvent : Nat := 188204
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188202 .coefficient, .predecessor 1 188203 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188202 .coefficient)
      LeftBound188200.bound (LeftBound188200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188203 .coefficient)
      LeftAuthority187799.bound (LeftAuthority187799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events733.exact187800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188200.bound, LeftAuthority187799.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188200.bound, LeftAuthority187799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188200.actual selector witness, LeftAuthority187799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188204

namespace LeftBound188208
def owner : Owner := ⟨.program ⟨257⟩, ⟨66819⟩⟩
def transferEvent : Nat := 188208
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188206 .coefficient, .predecessor 1 188207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188206 .coefficient)
      LeftBound188204.bound (LeftBound188204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188207 .coefficient)
      LeftAuthority187776.bound (LeftAuthority187776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events733.exact187777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187776.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188204.bound, LeftAuthority187776.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188204.bound, LeftAuthority187776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188204.actual selector witness, LeftAuthority187776.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188208

namespace LeftBound188212
def owner : Owner := ⟨.program ⟨257⟩, ⟨66820⟩⟩
def transferEvent : Nat := 188212
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188210 .coefficient, .predecessor 1 188211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188210 .coefficient)
      LeftBound188208.bound (LeftBound188208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188211 .coefficient)
      LeftAuthority187753.bound (LeftAuthority187753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events733.exact187754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187753.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187753.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188208.bound, LeftAuthority187753.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188208.bound, LeftAuthority187753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188208.actual selector witness, LeftAuthority187753.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188212

namespace LeftBound188215
def owner : Owner := ⟨.program ⟨257⟩, ⟨66821⟩⟩
def transferEvent : Nat := 188215
def frameStart : Nat := 187711
def rule : BoundRule := .identity (.predecessor 0 188214 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188214 .coefficient)
      LeftBound188212.bound (LeftBound188212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188212.derived selector witness)

def rawBound : CoeffClass := LeftBound188212.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound188212.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound188215

namespace LeftBound188232
def owner : Owner := ⟨.program ⟨257⟩, ⟨69099⟩⟩
def transferEvent : Nat := 188232
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188230 .coefficient, .predecessor 1 188231 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188230 .coefficient)
      LeftBound188215.bound (LeftBound188215.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound188215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188231 .coefficient)
      LeftAuthority188228.bound (LeftAuthority188228.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority188228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188215.bound, LeftAuthority188228.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188215.bound, LeftAuthority188228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188215.actual selector witness, LeftAuthority188228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188232

namespace LeftBound188235
def owner : Owner := ⟨.program ⟨257⟩, ⟨69100⟩⟩
def transferEvent : Nat := 188235
def frameStart : Nat := 187711
def rule : BoundRule := .identity (.predecessor 0 188234 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188234 .coefficient)
      LeftBound188232.bound (LeftBound188232.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound188232.derived selector witness)

def rawBound : CoeffClass := LeftBound188232.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound188232.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound188235

namespace LeftBound188241
def owner : Owner := ⟨.program ⟨257⟩, ⟨69101⟩⟩
def transferEvent : Nat := 188241
def frameStart : Nat := 187711
def rule : BoundRule := .product (.predecessor 0 188239 .coefficient) (.predecessor 1 188240 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188239 .coefficient)
      LeftAuthority188237.bound (LeftAuthority188237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188237.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188240 .coefficient)
      LeftBound188235.bound (LeftBound188235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority188237.bound LeftBound188235.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority188237.bound, LeftBound188235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority188237.actual selector witness) * (LeftBound188235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound188241

namespace LeftBound188317
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 188317
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188315 .coefficient, .predecessor 1 188316 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188315 .coefficient)
      LeftAuthority188313.bound (LeftAuthority188313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188313.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188316 .coefficient)
      LeftAuthority188310.bound (LeftAuthority188310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188310.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority188313.bound, LeftAuthority188310.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority188313.bound, LeftAuthority188310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority188313.actual selector witness, LeftAuthority188310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188317

namespace LeftBound188321
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 188321
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188319 .coefficient, .predecessor 1 188320 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188319 .coefficient)
      LeftBound188317.bound (LeftBound188317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188320 .coefficient)
      LeftAuthority188307.bound (LeftAuthority188307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188307.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188317.bound, LeftAuthority188307.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188317.bound, LeftAuthority188307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188317.actual selector witness, LeftAuthority188307.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188321

namespace LeftBound188325
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 188325
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188323 .coefficient, .predecessor 1 188324 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188323 .coefficient)
      LeftBound188321.bound (LeftBound188321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188324 .coefficient)
      LeftAuthority188304.bound (LeftAuthority188304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188304.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188304.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188321.bound, LeftAuthority188304.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188321.bound, LeftAuthority188304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188321.actual selector witness, LeftAuthority188304.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188325

namespace LeftBound188329
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 188329
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188327 .coefficient, .predecessor 1 188328 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188327 .coefficient)
      LeftBound188325.bound (LeftBound188325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188328 .coefficient)
      LeftAuthority188301.bound (LeftAuthority188301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188301.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188301.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188325.bound, LeftAuthority188301.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188325.bound, LeftAuthority188301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188325.actual selector witness, LeftAuthority188301.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188329

namespace LeftBound188333
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 188333
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188331 .coefficient, .predecessor 1 188332 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188331 .coefficient)
      LeftBound188329.bound (LeftBound188329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188329.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188332 .coefficient)
      LeftAuthority188298.bound (LeftAuthority188298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188298.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188298.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188329.bound, LeftAuthority188298.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188329.bound, LeftAuthority188298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188329.actual selector witness, LeftAuthority188298.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188333

namespace LeftBound188337
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 188337
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188335 .coefficient, .predecessor 1 188336 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188335 .coefficient)
      LeftBound188333.bound (LeftBound188333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188336 .coefficient)
      LeftAuthority188295.bound (LeftAuthority188295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188295.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188333.bound, LeftAuthority188295.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188333.bound, LeftAuthority188295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188333.actual selector witness, LeftAuthority188295.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188337

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
