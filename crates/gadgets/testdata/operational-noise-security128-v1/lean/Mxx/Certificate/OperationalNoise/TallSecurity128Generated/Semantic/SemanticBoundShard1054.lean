import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1053

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound158934
def owner : Owner := ⟨.program ⟨257⟩, ⟨66393⟩⟩
def transferEvent : Nat := 158934
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 158932 .coefficient, .predecessor 1 158933 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158932 .coefficient)
      LeftBound158930.bound (LeftBound158930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events620.exact158931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158933 .coefficient)
      LeftAuthority158664.bound (LeftAuthority158664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events619.exact158665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound158930.bound, LeftAuthority158664.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158930.bound, LeftAuthority158664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound158930.actual selector witness, LeftAuthority158664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound158934

namespace LeftBound158938
def owner : Owner := ⟨.program ⟨257⟩, ⟨66394⟩⟩
def transferEvent : Nat := 158938
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 158936 .coefficient, .predecessor 1 158937 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158936 .coefficient)
      LeftBound158934.bound (LeftBound158934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events620.exact158935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158937 .coefficient)
      LeftAuthority158641.bound (LeftAuthority158641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events619.exact158642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158641.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound158934.bound, LeftAuthority158641.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158934.bound, LeftAuthority158641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound158934.actual selector witness, LeftAuthority158641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound158938

namespace LeftBound158942
def owner : Owner := ⟨.program ⟨257⟩, ⟨66395⟩⟩
def transferEvent : Nat := 158942
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 158940 .coefficient, .predecessor 1 158941 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158940 .coefficient)
      LeftBound158938.bound (LeftBound158938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events620.exact158939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158941 .coefficient)
      LeftAuthority158618.bound (LeftAuthority158618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events619.exact158619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158618.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound158938.bound, LeftAuthority158618.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158938.bound, LeftAuthority158618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound158938.actual selector witness, LeftAuthority158618.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound158942

namespace LeftBound158946
def owner : Owner := ⟨.program ⟨257⟩, ⟨66396⟩⟩
def transferEvent : Nat := 158946
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 158944 .coefficient, .predecessor 1 158945 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158944 .coefficient)
      LeftBound158942.bound (LeftBound158942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events620.exact158943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158945 .coefficient)
      LeftAuthority158595.bound (LeftAuthority158595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events619.exact158596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158595.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound158942.bound, LeftAuthority158595.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158942.bound, LeftAuthority158595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound158942.actual selector witness, LeftAuthority158595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound158946

namespace LeftBound158950
def owner : Owner := ⟨.program ⟨257⟩, ⟨66397⟩⟩
def transferEvent : Nat := 158950
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 158948 .coefficient, .predecessor 1 158949 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158948 .coefficient)
      LeftBound158946.bound (LeftBound158946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events620.exact158947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158949 .coefficient)
      LeftAuthority158572.bound (LeftAuthority158572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events619.exact158573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158572.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound158946.bound, LeftAuthority158572.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158946.bound, LeftAuthority158572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound158946.actual selector witness, LeftAuthority158572.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound158950

namespace LeftBound158954
def owner : Owner := ⟨.program ⟨257⟩, ⟨66398⟩⟩
def transferEvent : Nat := 158954
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 158952 .coefficient, .predecessor 1 158953 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158952 .coefficient)
      LeftBound158950.bound (LeftBound158950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events620.exact158951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158950.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158953 .coefficient)
      LeftAuthority158549.bound (LeftAuthority158549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events619.exact158550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound158950.bound, LeftAuthority158549.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158950.bound, LeftAuthority158549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound158950.actual selector witness, LeftAuthority158549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound158954

namespace LeftBound158958
def owner : Owner := ⟨.program ⟨257⟩, ⟨66399⟩⟩
def transferEvent : Nat := 158958
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 158956 .coefficient, .predecessor 1 158957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158956 .coefficient)
      LeftBound158954.bound (LeftBound158954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events620.exact158955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158957 .coefficient)
      LeftAuthority158526.bound (LeftAuthority158526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events619.exact158527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158526.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound158954.bound, LeftAuthority158526.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158954.bound, LeftAuthority158526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound158954.actual selector witness, LeftAuthority158526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound158958

namespace LeftBound158962
def owner : Owner := ⟨.program ⟨257⟩, ⟨66400⟩⟩
def transferEvent : Nat := 158962
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 158960 .coefficient, .predecessor 1 158961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158960 .coefficient)
      LeftBound158958.bound (LeftBound158958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events620.exact158959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158961 .coefficient)
      LeftAuthority158503.bound (LeftAuthority158503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events619.exact158504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158503.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound158958.bound, LeftAuthority158503.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158958.bound, LeftAuthority158503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound158958.actual selector witness, LeftAuthority158503.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound158962

namespace LeftBound158965
def owner : Owner := ⟨.program ⟨257⟩, ⟨66401⟩⟩
def transferEvent : Nat := 158965
def frameStart : Nat := 158461
def rule : BoundRule := .identity (.predecessor 0 158964 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158964 .coefficient)
      LeftBound158962.bound (LeftBound158962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events620.exact158963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158962.derived selector witness)

def rawBound : CoeffClass := LeftBound158962.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound158962.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound158965

namespace LeftBound158982
def owner : Owner := ⟨.program ⟨257⟩, ⟨69075⟩⟩
def transferEvent : Nat := 158982
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 158980 .coefficient, .predecessor 1 158981 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158980 .coefficient)
      LeftBound158965.bound (LeftBound158965.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound158965.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158981 .coefficient)
      LeftAuthority158978.bound (LeftAuthority158978.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority158978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound158965.bound, LeftAuthority158978.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158965.bound, LeftAuthority158978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound158965.actual selector witness, LeftAuthority158978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound158982

namespace LeftBound158985
def owner : Owner := ⟨.program ⟨257⟩, ⟨69076⟩⟩
def transferEvent : Nat := 158985
def frameStart : Nat := 158461
def rule : BoundRule := .identity (.predecessor 0 158984 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158984 .coefficient)
      LeftBound158982.bound (LeftBound158982.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound158982.derived selector witness)

def rawBound : CoeffClass := LeftBound158982.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound158982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound158982.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound158985

namespace LeftBound158991
def owner : Owner := ⟨.program ⟨257⟩, ⟨69077⟩⟩
def transferEvent : Nat := 158991
def frameStart : Nat := 158461
def rule : BoundRule := .product (.predecessor 0 158989 .coefficient) (.predecessor 1 158990 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 158989 .coefficient)
      LeftAuthority158987.bound (LeftAuthority158987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact158988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158987.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 158990 .coefficient)
      LeftBound158985.bound (LeftBound158985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact158986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158985.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158985.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority158987.bound LeftBound158985.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority158987.bound, LeftBound158985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority158987.actual selector witness) * (LeftBound158985.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound158991

namespace LeftBound159067
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 159067
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159065 .coefficient, .predecessor 1 159066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159065 .coefficient)
      LeftAuthority159063.bound (LeftAuthority159063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159063.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159066 .coefficient)
      LeftAuthority159060.bound (LeftAuthority159060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159060.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159060.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority159063.bound, LeftAuthority159060.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority159063.bound, LeftAuthority159060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority159063.actual selector witness, LeftAuthority159060.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159067

namespace LeftBound159071
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 159071
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159069 .coefficient, .predecessor 1 159070 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159069 .coefficient)
      LeftBound159067.bound (LeftBound159067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159070 .coefficient)
      LeftAuthority159057.bound (LeftAuthority159057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159057.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159067.bound, LeftAuthority159057.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159067.bound, LeftAuthority159057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159067.actual selector witness, LeftAuthority159057.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159071

namespace LeftBound159075
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 159075
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159073 .coefficient, .predecessor 1 159074 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159073 .coefficient)
      LeftBound159071.bound (LeftBound159071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159074 .coefficient)
      LeftAuthority159054.bound (LeftAuthority159054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159054.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159071.bound, LeftAuthority159054.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159071.bound, LeftAuthority159054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159071.actual selector witness, LeftAuthority159054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159075

namespace LeftBound159079
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 159079
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159077 .coefficient, .predecessor 1 159078 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159077 .coefficient)
      LeftBound159075.bound (LeftBound159075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159078 .coefficient)
      LeftAuthority159051.bound (LeftAuthority159051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159051.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159051.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159075.bound, LeftAuthority159051.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159075.bound, LeftAuthority159051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159075.actual selector witness, LeftAuthority159051.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159079

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
