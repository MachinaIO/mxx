import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1729

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound255938
def owner : Owner := ⟨.program ⟨257⟩, ⟨64186⟩⟩
def transferEvent : Nat := 255938
def frameStart : Nat := 255888
def rule : BoundRule := .sum [.predecessor 0 255936 .coefficient, .predecessor 1 255937 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255936 .coefficient)
      LeftBound255921.bound (LeftBound255921.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound255921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 255937 .coefficient)
      LeftAuthority255934.bound (LeftAuthority255934.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority255934.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound255921.bound, LeftAuthority255934.bound]
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound255921.bound, LeftAuthority255934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound255921.actual selector witness, LeftAuthority255934.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound255938

namespace LeftBound255941
def owner : Owner := ⟨.program ⟨257⟩, ⟨64187⟩⟩
def transferEvent : Nat := 255941
def frameStart : Nat := 255888
def rule : BoundRule := .identity (.predecessor 0 255940 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255940 .coefficient)
      LeftBound255938.bound (LeftBound255938.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound255938.derived selector witness)

def rawBound : CoeffClass := LeftBound255938.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound255938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound255938.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound255941

namespace LeftBound255947
def owner : Owner := ⟨.program ⟨257⟩, ⟨64188⟩⟩
def transferEvent : Nat := 255947
def frameStart : Nat := 255888
def rule : BoundRule := .product (.predecessor 0 255945 .coefficient) (.predecessor 1 255946 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255945 .coefficient)
      LeftAuthority255943.bound (LeftAuthority255943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority255943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority255943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 255946 .coefficient)
      LeftBound255941.bound (LeftBound255941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255941.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority255943.bound LeftBound255941.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority255943.bound, LeftBound255941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority255943.actual selector witness) * (LeftBound255941.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound255947

namespace LeftBound255963
def owner : Owner := ⟨.program ⟨257⟩, ⟨9539⟩⟩
def transferEvent : Nat := 255963
def frameStart : Nat := 255888
def rule : BoundRule := .scale (.predecessor 0 255961 .coefficient) (.value (.predecessor 1 255962 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255961 .coefficient)
      LeftAuthority255959.bound (LeftAuthority255959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority255959.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority255959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 255962 .coefficient)
      LeftAuthority255950.bound (LeftAuthority255950.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority255950.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority255959.bound LeftAuthority255950.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority255959.bound, LeftAuthority255950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority255959.actual selector witness) * (LeftAuthority255950.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound255963

namespace LeftBound255966
def owner : Owner := ⟨.program ⟨257⟩, ⟨7293⟩⟩
def transferEvent : Nat := 255966
def frameStart : Nat := 255888
def rule : BoundRule := .identity (.predecessor 0 255965 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255965 .coefficient)
      LeftAuthority255953.bound (LeftAuthority255953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority255953.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority255953.derived selector witness)

def rawBound : CoeffClass := LeftAuthority255953.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority255953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority255953.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound255966

namespace LeftBound255970
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def transferEvent : Nat := 255970
def frameStart : Nat := 255888
def rule : BoundRule := .product (.predecessor 0 255968 .coefficient) (.predecessor 1 255969 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255968 .coefficient)
      LeftBound255966.bound (LeftBound255966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 255969 .coefficient)
      LeftBound255963.bound (LeftBound255963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255963.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound255966.bound LeftBound255963.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound255966.bound, LeftBound255963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound255966.actual selector witness) * (LeftBound255963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound255970

namespace LeftBound255975
def owner : Owner := ⟨.program ⟨257⟩, ⟨64189⟩⟩
def transferEvent : Nat := 255975
def frameStart : Nat := 255888
def rule : BoundRule := .sum [.predecessor 0 255973 .coefficient, .predecessor 1 255974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255973 .coefficient)
      LeftBound255970.bound (LeftBound255970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255970.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 255974 .coefficient)
      LeftBound255947.bound (LeftBound255947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound255970.bound, LeftBound255947.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound255970.bound, LeftBound255947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound255970.actual selector witness, LeftBound255947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound255975

namespace LeftBound255979
def owner : Owner := ⟨.program ⟨257⟩, ⟨64387⟩⟩
def transferEvent : Nat := 255979
def frameStart : Nat := 255888
def rule : BoundRule := .product (.predecessor 0 255977 .coefficient) (.predecessor 1 255978 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255977 .coefficient)
      LeftBound255975.bound (LeftBound255975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 255978 .coefficient)
      LeftAuthority255932.bound (LeftAuthority255932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority255932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority255932.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound255975.bound LeftAuthority255932.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound255975.bound, LeftAuthority255932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound255975.actual selector witness) * (LeftAuthority255932.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound255979

namespace LeftBound255990
def owner : Owner := ⟨.program ⟨257⟩, ⟨62770⟩⟩
def transferEvent : Nat := 255990
def frameStart : Nat := 255888
def rule : BoundRule := .product (.predecessor 0 255988 .coefficient) (.predecessor 1 255989 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255988 .coefficient)
      LeftAuthority255943.bound (LeftAuthority255943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority255943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority255943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 255989 .coefficient)
      LeftAuthority255986.bound (LeftAuthority255986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority255986.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority255986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority255943.bound LeftAuthority255986.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority255943.bound, LeftAuthority255986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority255943.actual selector witness) * (LeftAuthority255986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound255990

namespace LeftBound255998
def owner : Owner := ⟨.program ⟨257⟩, ⟨62771⟩⟩
def transferEvent : Nat := 255998
def frameStart : Nat := 255888
def rule : BoundRule := .sum [.predecessor 0 255996 .coefficient, .predecessor 1 255997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 255996 .coefficient)
      LeftAuthority255994.bound (LeftAuthority255994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority255994.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority255994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 255997 .coefficient)
      LeftBound255990.bound (LeftBound255990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255990.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority255994.bound, LeftBound255990.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority255994.bound, LeftBound255990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority255994.actual selector witness, LeftBound255990.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound255998

namespace LeftBound256002
def owner : Owner := ⟨.program ⟨257⟩, ⟨64388⟩⟩
def transferEvent : Nat := 256002
def frameStart : Nat := 255888
def rule : BoundRule := .sum [.predecessor 0 256000 .coefficient, .predecessor 1 256001 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 256000 .coefficient)
      LeftBound255998.bound (LeftBound255998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 256001 .coefficient)
      LeftBound255979.bound (LeftBound255979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound255998.bound, LeftBound255979.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound255998.bound, LeftBound255979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound255998.actual selector witness, LeftBound255979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound256002

namespace LeftBound256015
def owner : Owner := ⟨.program ⟨257⟩, ⟨64386⟩⟩
def transferEvent : Nat := 256015
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 256013 .coefficient, .predecessor 1 256014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 256013 .coefficient)
      LeftBound255836.bound (LeftBound255836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1000.exact256012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 256014 .coefficient)
      LeftBound255819.bound (LeftBound255819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events999.exact255826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255819.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound255836.bound, LeftBound255819.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound255836.bound, LeftBound255819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound255836.actual selector witness, LeftBound255819.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound256015

namespace LeftBound256018
def owner : Owner := ⟨.program ⟨257⟩, ⟨64386⟩⟩
def transferEvent : Nat := 256018
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 256012 .summary, .result 255826 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 256012 .summary)
      LeftBound255838.bound (LeftBound255838.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63322⟩⟩) (rawTerms := some (Proof.Events1000.exact256012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound255838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 255826 .summary)
      LeftBound255821.bound (LeftBound255821.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64385⟩⟩) (rawTerms := some (Proof.Events999.exact255826RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound255821.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound255838.bound, LeftBound255821.bound]
def bound : CoeffClass := .finite ⟨2997999239428004118528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound255838.bound, LeftBound255821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound255838.actual selector witness, LeftBound255821.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound256018

namespace LeftBound256022
def owner : Owner := ⟨.program ⟨257⟩, ⟨64719⟩⟩
def transferEvent : Nat := 256022
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 256020 .coefficient) (.predecessor 1 256021 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 256020 .coefficient)
      LeftBound256015.bound (LeftBound256015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1000.exact256019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound256015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound256015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 256021 .coefficient)
      LeftAuthority255741.bound (LeftAuthority255741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events998.exact255742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority255741.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority255741.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound256015.bound LeftAuthority255741.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound256015.bound, LeftAuthority255741.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound256015.actual selector witness) * (LeftAuthority255741.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound256022

namespace LeftBound256023
def owner : Owner := ⟨.program ⟨257⟩, ⟨64719⟩⟩
def transferEvent : Nat := 256023
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨64717⟩⟩]⟩ [⟨.result 255742 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 255742 .coefficient)
      LeftAuthority255741.bound (LeftAuthority255741.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨64717⟩⟩) (rawTerms := some (Proof.Events998.exact255742RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority255741.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority255741.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority255741.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority255741.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority255741.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound256023

namespace LeftBound256024
def owner : Owner := ⟨.program ⟨257⟩, ⟨64719⟩⟩
def transferEvent : Nat := 256024
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 256019 .summary) (.transfer 256023) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 256019 .summary)
      LeftBound256018.bound (LeftBound256018.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64386⟩⟩) (rawTerms := some (Proof.Events1000.exact256019RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound256018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 256023)
      LeftBound256023.bound (LeftBound256023.actual selector witness) := by
  exact .transfer (LeftBound256023.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound256018.bound LeftBound256023.bound
def bound : CoeffClass := .finite ⟨32190771716940378589077669150720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound256018.bound, LeftBound256023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound256018.actual selector witness) * (LeftBound256023.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound256024

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
