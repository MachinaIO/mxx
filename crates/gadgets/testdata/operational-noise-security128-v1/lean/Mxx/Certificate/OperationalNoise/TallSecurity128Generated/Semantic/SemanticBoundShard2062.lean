import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2040
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2044
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2047
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2055
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2058
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2061

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound302877
def owner : Owner := ⟨.program ⟨257⟩, ⟨15876⟩⟩
def transferEvent : Nat := 302877
def frameStart : Nat := 302801
def rule : BoundRule := .product (.predecessor 0 302875 .coefficient) (.predecessor 1 302876 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302875 .coefficient)
      LeftAuthority302850.bound (LeftAuthority302850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302850.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302876 .coefficient)
      LeftAuthority302873.bound (LeftAuthority302873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302873.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302873.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority302850.bound LeftAuthority302873.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302850.bound, LeftAuthority302873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority302850.actual selector witness) * (LeftAuthority302873.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302877

namespace LeftBound302885
def owner : Owner := ⟨.program ⟨257⟩, ⟨15877⟩⟩
def transferEvent : Nat := 302885
def frameStart : Nat := 302801
def rule : BoundRule := .sum [.predecessor 0 302883 .coefficient, .predecessor 1 302884 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302883 .coefficient)
      LeftAuthority302881.bound (LeftAuthority302881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302884 .coefficient)
      LeftBound302877.bound (LeftBound302877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302877.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302877.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority302881.bound, LeftBound302877.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302881.bound, LeftBound302877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority302881.actual selector witness, LeftBound302877.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302885

namespace LeftBound302889
def owner : Owner := ⟨.program ⟨257⟩, ⟨17485⟩⟩
def transferEvent : Nat := 302889
def frameStart : Nat := 302801
def rule : BoundRule := .sum [.predecessor 0 302887 .coefficient, .predecessor 1 302888 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302887 .coefficient)
      LeftBound302885.bound (LeftBound302885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302885.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302885.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302888 .coefficient)
      LeftBound302866.bound (LeftBound302866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302885.bound, LeftBound302866.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302885.bound, LeftBound302866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302885.actual selector witness, LeftBound302866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302889

namespace LeftBound302902
def owner : Owner := ⟨.program ⟨257⟩, ⟨17484⟩⟩
def transferEvent : Nat := 302902
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302900 .coefficient, .predecessor 1 302901 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302900 .coefficient)
      LeftBound302755.bound (LeftBound302755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302901 .coefficient)
      LeftBound302738.bound (LeftBound302738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1182.exact302745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302738.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302755.bound, LeftBound302738.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302755.bound, LeftBound302738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302755.actual selector witness, LeftBound302738.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302902

namespace LeftBound302905
def owner : Owner := ⟨.program ⟨257⟩, ⟨17484⟩⟩
def transferEvent : Nat := 302905
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302899 .summary, .result 302745 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302899 .summary)
      LeftBound302757.bound (LeftBound302757.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16399⟩⟩) (rawTerms := some (Proof.Events1183.exact302899RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302757.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302745 .summary)
      LeftBound302740.bound (LeftBound302740.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17483⟩⟩) (rawTerms := some (Proof.Events1182.exact302745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302740.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302757.bound, LeftBound302740.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302757.bound, LeftBound302740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302757.actual selector witness, LeftBound302740.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302905

namespace LeftBound302909
def owner : Owner := ⟨.program ⟨257⟩, ⟨20346⟩⟩
def transferEvent : Nat := 302909
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302907 .coefficient, .predecessor 1 302908 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302907 .coefficient)
      LeftBound302902.bound (LeftBound302902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302908 .coefficient)
      LeftBound302468.bound (LeftBound302468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302468.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302902.bound, LeftBound302468.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302902.bound, LeftBound302468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302902.actual selector witness, LeftBound302468.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302909

namespace LeftBound302910
def owner : Owner := ⟨.program ⟨257⟩, ⟨20346⟩⟩
def transferEvent : Nat := 302910
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302906 .summary, .result 302472 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302906 .summary)
      LeftBound302905.bound (LeftBound302905.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17484⟩⟩) (rawTerms := some (Proof.Events1183.exact302906RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302905.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302472 .summary)
      LeftBound302471.bound (LeftBound302471.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20345⟩⟩) (rawTerms := some (Proof.Events1181.exact302472RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302471.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302905.bound, LeftBound302471.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302905.bound, LeftBound302471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302905.actual selector witness, LeftBound302471.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302910

namespace LeftBound302914
def owner : Owner := ⟨.program ⟨257⟩, ⟨23566⟩⟩
def transferEvent : Nat := 302914
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302912 .coefficient, .predecessor 1 302913 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302912 .coefficient)
      LeftBound302909.bound (LeftBound302909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302913 .coefficient)
      LeftBound302034.bound (LeftBound302034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact302038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302034.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302034.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302909.bound, LeftBound302034.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302909.bound, LeftBound302034.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302909.actual selector witness, LeftBound302034.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302914

namespace LeftBound302915
def owner : Owner := ⟨.program ⟨257⟩, ⟨23566⟩⟩
def transferEvent : Nat := 302915
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302911 .summary, .result 302038 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302911 .summary)
      LeftBound302910.bound (LeftBound302910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20346⟩⟩) (rawTerms := some (Proof.Events1183.exact302911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302038 .summary)
      LeftBound302037.bound (LeftBound302037.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23565⟩⟩) (rawTerms := some (Proof.Events1179.exact302038RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302037.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302910.bound, LeftBound302037.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302910.bound, LeftBound302037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302910.actual selector witness, LeftBound302037.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302915

namespace LeftBound302919
def owner : Owner := ⟨.program ⟨257⟩, ⟨33586⟩⟩
def transferEvent : Nat := 302919
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302917 .coefficient, .predecessor 1 302918 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302917 .coefficient)
      LeftBound302914.bound (LeftBound302914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302918 .coefficient)
      LeftBound301600.bound (LeftBound301600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301600.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301600.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302914.bound, LeftBound301600.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302914.bound, LeftBound301600.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302914.actual selector witness, LeftBound301600.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302919

namespace LeftBound302920
def owner : Owner := ⟨.program ⟨257⟩, ⟨33586⟩⟩
def transferEvent : Nat := 302920
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302916 .summary, .result 301604 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302916 .summary)
      LeftBound302915.bound (LeftBound302915.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23566⟩⟩) (rawTerms := some (Proof.Events1183.exact302916RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301604 .summary)
      LeftBound301603.bound (LeftBound301603.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33585⟩⟩) (rawTerms := some (Proof.Events1178.exact301604RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302915.bound, LeftBound301603.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302915.bound, LeftBound301603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302915.actual selector witness, LeftBound301603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302920

namespace LeftBound302924
def owner : Owner := ⟨.program ⟨257⟩, ⟨52646⟩⟩
def transferEvent : Nat := 302924
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302922 .coefficient, .predecessor 1 302923 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302922 .coefficient)
      LeftBound302919.bound (LeftBound302919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302919.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302923 .coefficient)
      LeftBound301166.bound (LeftBound301166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302919.bound, LeftBound301166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302919.bound, LeftBound301166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302919.actual selector witness, LeftBound301166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302924

namespace LeftBound302925
def owner : Owner := ⟨.program ⟨257⟩, ⟨52646⟩⟩
def transferEvent : Nat := 302925
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302921 .summary, .result 301170 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302921 .summary)
      LeftBound302920.bound (LeftBound302920.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33586⟩⟩) (rawTerms := some (Proof.Events1183.exact302921RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301170 .summary)
      LeftBound301169.bound (LeftBound301169.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52645⟩⟩) (rawTerms := some (Proof.Events1176.exact301170RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301169.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302920.bound, LeftBound301169.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302920.bound, LeftBound301169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302920.actual selector witness, LeftBound301169.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302925

namespace LeftBound302929
def owner : Owner := ⟨.program ⟨257⟩, ⟨55626⟩⟩
def transferEvent : Nat := 302929
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302927 .coefficient, .predecessor 1 302928 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302927 .coefficient)
      LeftBound302924.bound (LeftBound302924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302928 .coefficient)
      LeftBound300732.bound (LeftBound300732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1174.exact300736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound300732.bound, RecordedBoundRefines] <;> decide)
      (LeftBound300732.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302924.bound, LeftBound300732.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302924.bound, LeftBound300732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302924.actual selector witness, LeftBound300732.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302929

namespace LeftBound302930
def owner : Owner := ⟨.program ⟨257⟩, ⟨55626⟩⟩
def transferEvent : Nat := 302930
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302926 .summary, .result 300736 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302926 .summary)
      LeftBound302925.bound (LeftBound302925.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52646⟩⟩) (rawTerms := some (Proof.Events1183.exact302926RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 300736 .summary)
      LeftBound300735.bound (LeftBound300735.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55625⟩⟩) (rawTerms := some (Proof.Events1174.exact300736RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound300735.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302925.bound, LeftBound300735.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302925.bound, LeftBound300735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302925.actual selector witness, LeftBound300735.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302930

namespace LeftBound302934
def owner : Owner := ⟨.program ⟨257⟩, ⟨58606⟩⟩
def transferEvent : Nat := 302934
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302932 .coefficient, .predecessor 1 302933 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302932 .coefficient)
      LeftBound302929.bound (LeftBound302929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302929.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302929.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302933 .coefficient)
      LeftBound300298.bound (LeftBound300298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1173.exact300302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound300298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound300298.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302929.bound, LeftBound300298.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302929.bound, LeftBound300298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302929.actual selector witness, LeftBound300298.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302934

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
