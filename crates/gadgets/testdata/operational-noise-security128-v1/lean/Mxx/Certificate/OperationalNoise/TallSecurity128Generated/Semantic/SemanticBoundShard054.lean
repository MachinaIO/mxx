import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound15895
def owner : Owner := ⟨.program ⟨257⟩, ⟨7292⟩⟩
def transferEvent : Nat := 15895
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 15894 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15894 .coefficient)
      LeftAuthority15892.bound (LeftAuthority15892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15892.derived selector witness)

def rawBound : CoeffClass := LeftAuthority15892.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority15892.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound15895

namespace LeftBound15899
def owner : Owner := ⟨.program ⟨257⟩, ⟨9128⟩⟩
def transferEvent : Nat := 15899
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15897 .coefficient, .predecessor 1 15898 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15897 .coefficient)
      LeftBound15895.bound (LeftBound15895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15898 .coefficient)
      LeftBound15895.bound (LeftBound15895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15895.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15895.bound, LeftBound15895.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15895.bound, LeftBound15895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15895.actual selector witness, LeftBound15895.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15899

namespace LeftBound15904
def owner : Owner := ⟨.program ⟨257⟩, ⟨9129⟩⟩
def transferEvent : Nat := 15904
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15902 .coefficient, .predecessor 1 15903 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15902 .coefficient)
      LeftBound15899.bound (LeftBound15899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15899.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15903 .coefficient)
      LeftBound15888.bound (LeftBound15888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15888.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15899.bound, LeftBound15888.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15899.bound, LeftBound15888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15899.actual selector witness, LeftBound15888.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15904

namespace LeftBound15908
def owner : Owner := ⟨.program ⟨257⟩, ⟨9130⟩⟩
def transferEvent : Nat := 15908
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15906 .coefficient, .predecessor 1 15907 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15906 .coefficient)
      LeftBound15904.bound (LeftBound15904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15907 .coefficient)
      LeftBound15868.bound (LeftBound15868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15868.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15904.bound, LeftBound15868.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15904.bound, LeftBound15868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15904.actual selector witness, LeftBound15868.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15908

namespace LeftBound15912
def owner : Owner := ⟨.program ⟨257⟩, ⟨9131⟩⟩
def transferEvent : Nat := 15912
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15910 .coefficient, .predecessor 1 15911 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15910 .coefficient)
      LeftBound15908.bound (LeftBound15908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15911 .coefficient)
      LeftBound15848.bound (LeftBound15848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15848.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15908.bound, LeftBound15848.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15908.bound, LeftBound15848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15908.actual selector witness, LeftBound15848.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15912

namespace LeftBound15916
def owner : Owner := ⟨.program ⟨257⟩, ⟨9132⟩⟩
def transferEvent : Nat := 15916
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15914 .coefficient, .predecessor 1 15915 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15914 .coefficient)
      LeftBound15912.bound (LeftBound15912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15915 .coefficient)
      LeftBound15828.bound (LeftBound15828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15828.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15912.bound, LeftBound15828.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15912.bound, LeftBound15828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15912.actual selector witness, LeftBound15828.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15916

namespace LeftBound15920
def owner : Owner := ⟨.program ⟨257⟩, ⟨9133⟩⟩
def transferEvent : Nat := 15920
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15918 .coefficient, .predecessor 1 15919 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15918 .coefficient)
      LeftBound15916.bound (LeftBound15916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15916.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15919 .coefficient)
      LeftBound15808.bound (LeftBound15808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15916.bound, LeftBound15808.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15916.bound, LeftBound15808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15916.actual selector witness, LeftBound15808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15920

namespace LeftBound15924
def owner : Owner := ⟨.program ⟨257⟩, ⟨9134⟩⟩
def transferEvent : Nat := 15924
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15922 .coefficient, .predecessor 1 15923 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15922 .coefficient)
      LeftBound15920.bound (LeftBound15920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15923 .coefficient)
      LeftBound15788.bound (LeftBound15788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15788.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15788.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15920.bound, LeftBound15788.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15920.bound, LeftBound15788.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15920.actual selector witness, LeftBound15788.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15924

namespace LeftBound15928
def owner : Owner := ⟨.program ⟨257⟩, ⟨9135⟩⟩
def transferEvent : Nat := 15928
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15926 .coefficient, .predecessor 1 15927 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15926 .coefficient)
      LeftBound15924.bound (LeftBound15924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15927 .coefficient)
      LeftBound15768.bound (LeftBound15768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15768.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15768.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15924.bound, LeftBound15768.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15924.bound, LeftBound15768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15924.actual selector witness, LeftBound15768.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15928

namespace LeftBound15932
def owner : Owner := ⟨.program ⟨257⟩, ⟨9136⟩⟩
def transferEvent : Nat := 15932
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15930 .coefficient, .predecessor 1 15931 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15930 .coefficient)
      LeftBound15928.bound (LeftBound15928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15928.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15931 .coefficient)
      LeftBound15748.bound (LeftBound15748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15748.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15928.bound, LeftBound15748.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15928.bound, LeftBound15748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15928.actual selector witness, LeftBound15748.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15932

namespace LeftBound15936
def owner : Owner := ⟨.program ⟨257⟩, ⟨9137⟩⟩
def transferEvent : Nat := 15936
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15934 .coefficient, .predecessor 1 15935 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15934 .coefficient)
      LeftBound15932.bound (LeftBound15932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15932.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15935 .coefficient)
      LeftBound15728.bound (LeftBound15728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15932.bound, LeftBound15728.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15932.bound, LeftBound15728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15932.actual selector witness, LeftBound15728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15936

namespace LeftBound15940
def owner : Owner := ⟨.program ⟨257⟩, ⟨9138⟩⟩
def transferEvent : Nat := 15940
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15938 .coefficient, .predecessor 1 15939 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15938 .coefficient)
      LeftBound15936.bound (LeftBound15936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15939 .coefficient)
      LeftBound15708.bound (LeftBound15708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15708.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15936.bound, LeftBound15708.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15936.bound, LeftBound15708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15936.actual selector witness, LeftBound15708.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15940

namespace LeftBound15944
def owner : Owner := ⟨.program ⟨257⟩, ⟨9139⟩⟩
def transferEvent : Nat := 15944
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15942 .coefficient, .predecessor 1 15943 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15942 .coefficient)
      LeftBound15940.bound (LeftBound15940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15940.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15943 .coefficient)
      LeftBound15688.bound (LeftBound15688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15688.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15940.bound, LeftBound15688.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15940.bound, LeftBound15688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15940.actual selector witness, LeftBound15688.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15944

namespace LeftBound15948
def owner : Owner := ⟨.program ⟨257⟩, ⟨9140⟩⟩
def transferEvent : Nat := 15948
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15946 .coefficient, .predecessor 1 15947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15946 .coefficient)
      LeftBound15944.bound (LeftBound15944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15947 .coefficient)
      LeftBound15668.bound (LeftBound15668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15944.bound, LeftBound15668.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15944.bound, LeftBound15668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15944.actual selector witness, LeftBound15668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15948

namespace LeftBound15952
def owner : Owner := ⟨.program ⟨257⟩, ⟨9141⟩⟩
def transferEvent : Nat := 15952
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15950 .coefficient, .predecessor 1 15951 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15950 .coefficient)
      LeftBound15948.bound (LeftBound15948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15951 .coefficient)
      LeftBound15648.bound (LeftBound15648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15948.bound, LeftBound15648.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15948.bound, LeftBound15648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15948.actual selector witness, LeftBound15648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15952

namespace LeftBound15956
def owner : Owner := ⟨.program ⟨257⟩, ⟨9142⟩⟩
def transferEvent : Nat := 15956
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15954 .coefficient, .predecessor 1 15955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15954 .coefficient)
      LeftBound15952.bound (LeftBound15952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15955 .coefficient)
      LeftBound15628.bound (LeftBound15628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15952.bound, LeftBound15628.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15952.bound, LeftBound15628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15952.actual selector witness, LeftBound15628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15956

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
