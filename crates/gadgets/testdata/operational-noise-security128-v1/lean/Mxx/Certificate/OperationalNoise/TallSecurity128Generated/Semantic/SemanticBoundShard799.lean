import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard798

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound121906
def owner : Owner := ⟨.program ⟨257⟩, ⟨38691⟩⟩
def transferEvent : Nat := 121906
def frameStart : Nat := 121853
def rule : BoundRule := .identity (.predecessor 0 121905 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121905 .coefficient)
      LeftBound121903.bound (LeftBound121903.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound121903.derived selector witness)

def rawBound : CoeffClass := LeftBound121903.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound121903.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound121906

namespace LeftBound121912
def owner : Owner := ⟨.program ⟨257⟩, ⟨38692⟩⟩
def transferEvent : Nat := 121912
def frameStart : Nat := 121853
def rule : BoundRule := .product (.predecessor 0 121910 .coefficient) (.predecessor 1 121911 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121910 .coefficient)
      LeftAuthority121908.bound (LeftAuthority121908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121911 .coefficient)
      LeftBound121906.bound (LeftBound121906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121906.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority121908.bound LeftBound121906.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority121908.bound, LeftBound121906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority121908.actual selector witness) * (LeftBound121906.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound121912

namespace LeftBound121928
def owner : Owner := ⟨.program ⟨257⟩, ⟨9554⟩⟩
def transferEvent : Nat := 121928
def frameStart : Nat := 121853
def rule : BoundRule := .scale (.predecessor 0 121926 .coefficient) (.value (.predecessor 1 121927 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121926 .coefficient)
      LeftAuthority121924.bound (LeftAuthority121924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121927 .coefficient)
      LeftAuthority121915.bound (LeftAuthority121915.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority121915.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority121924.bound LeftAuthority121915.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority121924.bound, LeftAuthority121915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority121924.actual selector witness) * (LeftAuthority121915.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound121928

namespace LeftBound121931
def owner : Owner := ⟨.program ⟨257⟩, ⟨7298⟩⟩
def transferEvent : Nat := 121931
def frameStart : Nat := 121853
def rule : BoundRule := .identity (.predecessor 0 121930 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121930 .coefficient)
      LeftAuthority121918.bound (LeftAuthority121918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121918.derived selector witness)

def rawBound : CoeffClass := LeftAuthority121918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority121918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority121918.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound121931

namespace LeftBound121935
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def transferEvent : Nat := 121935
def frameStart : Nat := 121853
def rule : BoundRule := .product (.predecessor 0 121933 .coefficient) (.predecessor 1 121934 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121933 .coefficient)
      LeftBound121931.bound (LeftBound121931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121934 .coefficient)
      LeftBound121928.bound (LeftBound121928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121928.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound121931.bound LeftBound121928.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121931.bound, LeftBound121928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound121931.actual selector witness) * (LeftBound121928.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound121935

namespace LeftBound121940
def owner : Owner := ⟨.program ⟨257⟩, ⟨38693⟩⟩
def transferEvent : Nat := 121940
def frameStart : Nat := 121853
def rule : BoundRule := .sum [.predecessor 0 121938 .coefficient, .predecessor 1 121939 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121938 .coefficient)
      LeftBound121935.bound (LeftBound121935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121939 .coefficient)
      LeftBound121912.bound (LeftBound121912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121912.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound121935.bound, LeftBound121912.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121935.bound, LeftBound121912.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound121935.actual selector witness, LeftBound121912.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound121940

namespace LeftBound121944
def owner : Owner := ⟨.program ⟨257⟩, ⟨38898⟩⟩
def transferEvent : Nat := 121944
def frameStart : Nat := 121853
def rule : BoundRule := .product (.predecessor 0 121942 .coefficient) (.predecessor 1 121943 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121942 .coefficient)
      LeftBound121940.bound (LeftBound121940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121940.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121943 .coefficient)
      LeftAuthority121897.bound (LeftAuthority121897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121897.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121897.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound121940.bound LeftAuthority121897.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121940.bound, LeftAuthority121897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound121940.actual selector witness) * (LeftAuthority121897.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound121944

namespace LeftBound121955
def owner : Owner := ⟨.program ⟨257⟩, ⟨37398⟩⟩
def transferEvent : Nat := 121955
def frameStart : Nat := 121853
def rule : BoundRule := .product (.predecessor 0 121953 .coefficient) (.predecessor 1 121954 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121953 .coefficient)
      LeftAuthority121908.bound (LeftAuthority121908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121954 .coefficient)
      LeftAuthority121951.bound (LeftAuthority121951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121951.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121951.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority121908.bound LeftAuthority121951.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority121908.bound, LeftAuthority121951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority121908.actual selector witness) * (LeftAuthority121951.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound121955

namespace LeftBound121963
def owner : Owner := ⟨.program ⟨257⟩, ⟨37399⟩⟩
def transferEvent : Nat := 121963
def frameStart : Nat := 121853
def rule : BoundRule := .sum [.predecessor 0 121961 .coefficient, .predecessor 1 121962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121961 .coefficient)
      LeftAuthority121959.bound (LeftAuthority121959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121959.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121962 .coefficient)
      LeftBound121955.bound (LeftBound121955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121955.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121955.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority121959.bound, LeftBound121955.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority121959.bound, LeftBound121955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority121959.actual selector witness, LeftBound121955.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound121963

namespace LeftBound121967
def owner : Owner := ⟨.program ⟨257⟩, ⟨38899⟩⟩
def transferEvent : Nat := 121967
def frameStart : Nat := 121853
def rule : BoundRule := .sum [.predecessor 0 121965 .coefficient, .predecessor 1 121966 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121965 .coefficient)
      LeftBound121963.bound (LeftBound121963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121966 .coefficient)
      LeftBound121944.bound (LeftBound121944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121944.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound121963.bound, LeftBound121944.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121963.bound, LeftBound121944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound121963.actual selector witness, LeftBound121944.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound121967

namespace LeftBound121980
def owner : Owner := ⟨.program ⟨257⟩, ⟨38897⟩⟩
def transferEvent : Nat := 121980
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 121978 .coefficient, .predecessor 1 121979 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121978 .coefficient)
      LeftBound121801.bound (LeftBound121801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121979 .coefficient)
      LeftBound121784.bound (LeftBound121784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events475.exact121791RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121784.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121784.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound121801.bound, LeftBound121784.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121801.bound, LeftBound121784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound121801.actual selector witness, LeftBound121784.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound121980

namespace LeftBound121983
def owner : Owner := ⟨.program ⟨257⟩, ⟨38897⟩⟩
def transferEvent : Nat := 121983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 121977 .summary, .result 121791 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 121977 .summary)
      LeftBound121803.bound (LeftBound121803.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37832⟩⟩) (rawTerms := some (Proof.Events476.exact121977RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound121803.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 121791 .summary)
      LeftBound121786.bound (LeftBound121786.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38896⟩⟩) (rawTerms := some (Proof.Events475.exact121791RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound121786.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound121803.bound, LeftBound121786.bound]
def bound : CoeffClass := .finite ⟨2998182198162866044928, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121803.bound, LeftBound121786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound121803.actual selector witness, LeftBound121786.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound121983

namespace LeftBound121987
def owner : Owner := ⟨.program ⟨257⟩, ⟨39211⟩⟩
def transferEvent : Nat := 121987
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 121985 .coefficient) (.predecessor 1 121986 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121985 .coefficient)
      LeftBound121980.bound (LeftBound121980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121986 .coefficient)
      LeftAuthority121706.bound (LeftAuthority121706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events475.exact121707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121706.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound121980.bound LeftAuthority121706.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121980.bound, LeftAuthority121706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound121980.actual selector witness) * (LeftAuthority121706.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound121987

namespace LeftBound121988
def owner : Owner := ⟨.program ⟨257⟩, ⟨39211⟩⟩
def transferEvent : Nat := 121988
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨39209⟩⟩]⟩ [⟨.result 121707 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 121707 .coefficient)
      LeftAuthority121706.bound (LeftAuthority121706.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨39209⟩⟩) (rawTerms := some (Proof.Events475.exact121707RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121706.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority121706.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority121706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority121706.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound121988

namespace LeftBound121989
def owner : Owner := ⟨.program ⟨257⟩, ⟨39211⟩⟩
def transferEvent : Nat := 121989
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 121984 .summary) (.transfer 121988) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 121984 .summary)
      LeftBound121983.bound (LeftBound121983.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38897⟩⟩) (rawTerms := some (Proof.Events476.exact121984RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound121983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 121988)
      LeftBound121988.bound (LeftBound121988.actual selector witness) := by
  exact .transfer (LeftBound121988.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound121983.bound LeftBound121988.bound
def bound : CoeffClass := .finite ⟨32192736221397252361486566686720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121983.bound, LeftBound121988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound121983.actual selector witness) * (LeftBound121988.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound121989

namespace LeftBound122000
def owner : Owner := ⟨.program ⟨257⟩, ⟨38098⟩⟩
def transferEvent : Nat := 122000
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 121998 .coefficient) (.value (.predecessor 1 121999 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 121998 .coefficient)
      LeftAuthority121996.bound (LeftAuthority121996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority121996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority121996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 121999 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority121996.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority121996.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority121996.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound122000

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
