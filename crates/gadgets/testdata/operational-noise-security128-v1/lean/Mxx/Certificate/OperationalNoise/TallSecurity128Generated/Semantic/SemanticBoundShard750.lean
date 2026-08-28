import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard749

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound115079
def owner : Owner := ⟨.program ⟨257⟩, ⟨66678⟩⟩
def transferEvent : Nat := 115079
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115077 .coefficient, .predecessor 1 115078 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115077 .coefficient)
      LeftBound115075.bound (LeftBound115075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115078 .coefficient)
      LeftAuthority114674.bound (LeftAuthority114674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events447.exact114675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114674.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115075.bound, LeftAuthority114674.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115075.bound, LeftAuthority114674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115075.actual selector witness, LeftAuthority114674.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115079

namespace LeftBound115083
def owner : Owner := ⟨.program ⟨257⟩, ⟨66679⟩⟩
def transferEvent : Nat := 115083
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115081 .coefficient, .predecessor 1 115082 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115081 .coefficient)
      LeftBound115079.bound (LeftBound115079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115082 .coefficient)
      LeftAuthority114651.bound (LeftAuthority114651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events447.exact114652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114651.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114651.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115079.bound, LeftAuthority114651.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115079.bound, LeftAuthority114651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115079.actual selector witness, LeftAuthority114651.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115083

namespace LeftBound115087
def owner : Owner := ⟨.program ⟨257⟩, ⟨66680⟩⟩
def transferEvent : Nat := 115087
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115085 .coefficient, .predecessor 1 115086 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115085 .coefficient)
      LeftBound115083.bound (LeftBound115083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115086 .coefficient)
      LeftAuthority114628.bound (LeftAuthority114628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events447.exact114629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114628.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115083.bound, LeftAuthority114628.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115083.bound, LeftAuthority114628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115083.actual selector witness, LeftAuthority114628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115087

namespace LeftBound115090
def owner : Owner := ⟨.program ⟨257⟩, ⟨66681⟩⟩
def transferEvent : Nat := 115090
def frameStart : Nat := 114586
def rule : BoundRule := .identity (.predecessor 0 115089 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115089 .coefficient)
      LeftBound115087.bound (LeftBound115087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115087.derived selector witness)

def rawBound : CoeffClass := LeftBound115087.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound115087.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound115090

namespace LeftBound115107
def owner : Owner := ⟨.program ⟨257⟩, ⟨69091⟩⟩
def transferEvent : Nat := 115107
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115105 .coefficient, .predecessor 1 115106 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115105 .coefficient)
      LeftBound115090.bound (LeftBound115090.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound115090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115106 .coefficient)
      LeftAuthority115103.bound (LeftAuthority115103.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority115103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115090.bound, LeftAuthority115103.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115090.bound, LeftAuthority115103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115090.actual selector witness, LeftAuthority115103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115107

namespace LeftBound115110
def owner : Owner := ⟨.program ⟨257⟩, ⟨69092⟩⟩
def transferEvent : Nat := 115110
def frameStart : Nat := 114586
def rule : BoundRule := .identity (.predecessor 0 115109 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115109 .coefficient)
      LeftBound115107.bound (LeftBound115107.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound115107.derived selector witness)

def rawBound : CoeffClass := LeftBound115107.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound115107.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound115110

namespace LeftBound115116
def owner : Owner := ⟨.program ⟨257⟩, ⟨69093⟩⟩
def transferEvent : Nat := 115116
def frameStart : Nat := 114586
def rule : BoundRule := .product (.predecessor 0 115114 .coefficient) (.predecessor 1 115115 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115114 .coefficient)
      LeftAuthority115112.bound (LeftAuthority115112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115112.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115115 .coefficient)
      LeftBound115110.bound (LeftBound115110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115110.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority115112.bound LeftBound115110.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115112.bound, LeftBound115110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority115112.actual selector witness) * (LeftBound115110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115116

namespace LeftBound115192
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 115192
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115190 .coefficient, .predecessor 1 115191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115190 .coefficient)
      LeftAuthority115188.bound (LeftAuthority115188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115188.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115191 .coefficient)
      LeftAuthority115185.bound (LeftAuthority115185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115185.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115185.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority115188.bound, LeftAuthority115185.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115188.bound, LeftAuthority115185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority115188.actual selector witness, LeftAuthority115185.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115192

namespace LeftBound115196
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 115196
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115194 .coefficient, .predecessor 1 115195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115194 .coefficient)
      LeftBound115192.bound (LeftBound115192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115195 .coefficient)
      LeftAuthority115182.bound (LeftAuthority115182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115182.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115192.bound, LeftAuthority115182.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115192.bound, LeftAuthority115182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115192.actual selector witness, LeftAuthority115182.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115196

namespace LeftBound115200
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 115200
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115198 .coefficient, .predecessor 1 115199 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115198 .coefficient)
      LeftBound115196.bound (LeftBound115196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115199 .coefficient)
      LeftAuthority115179.bound (LeftAuthority115179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115179.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115196.bound, LeftAuthority115179.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115196.bound, LeftAuthority115179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115196.actual selector witness, LeftAuthority115179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115200

namespace LeftBound115204
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 115204
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115202 .coefficient, .predecessor 1 115203 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115202 .coefficient)
      LeftBound115200.bound (LeftBound115200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115203 .coefficient)
      LeftAuthority115176.bound (LeftAuthority115176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115200.bound, LeftAuthority115176.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115200.bound, LeftAuthority115176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115200.actual selector witness, LeftAuthority115176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115204

namespace LeftBound115208
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 115208
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115206 .coefficient, .predecessor 1 115207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115206 .coefficient)
      LeftBound115204.bound (LeftBound115204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115207 .coefficient)
      LeftAuthority115173.bound (LeftAuthority115173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115173.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115204.bound, LeftAuthority115173.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115204.bound, LeftAuthority115173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115204.actual selector witness, LeftAuthority115173.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115208

namespace LeftBound115212
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 115212
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115210 .coefficient, .predecessor 1 115211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115210 .coefficient)
      LeftBound115208.bound (LeftBound115208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115211 .coefficient)
      LeftAuthority115170.bound (LeftAuthority115170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115170.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115170.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115208.bound, LeftAuthority115170.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115208.bound, LeftAuthority115170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115208.actual selector witness, LeftAuthority115170.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115212

namespace LeftBound115216
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 115216
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115214 .coefficient, .predecessor 1 115215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115214 .coefficient)
      LeftBound115212.bound (LeftBound115212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115215 .coefficient)
      LeftAuthority115167.bound (LeftAuthority115167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115212.bound, LeftAuthority115167.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115212.bound, LeftAuthority115167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115212.actual selector witness, LeftAuthority115167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115216

namespace LeftBound115220
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 115220
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115218 .coefficient, .predecessor 1 115219 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115218 .coefficient)
      LeftBound115216.bound (LeftBound115216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115219 .coefficient)
      LeftAuthority115164.bound (LeftAuthority115164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115164.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115216.bound, LeftAuthority115164.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115216.bound, LeftAuthority115164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115216.actual selector witness, LeftAuthority115164.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115220

namespace LeftBound115224
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 115224
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115222 .coefficient, .predecessor 1 115223 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115222 .coefficient)
      LeftBound115220.bound (LeftBound115220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115220.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115223 .coefficient)
      LeftAuthority115161.bound (LeftAuthority115161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115161.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115220.bound, LeftAuthority115161.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115220.bound, LeftAuthority115161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115220.actual selector witness, LeftAuthority115161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115224

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
