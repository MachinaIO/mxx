import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard241

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound41902
def owner : Owner := ⟨.program ⟨257⟩, ⟨22258⟩⟩
def transferEvent : Nat := 41902
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41900 .coefficient, .predecessor 1 41901 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41900 .coefficient)
      LeftBound41898.bound (LeftBound41898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41901 .coefficient)
      LeftAuthority41848.bound (LeftAuthority41848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41848.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41848.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41898.bound, LeftAuthority41848.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41898.bound, LeftAuthority41848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41898.actual selector witness, LeftAuthority41848.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41902

namespace LeftBound41906
def owner : Owner := ⟨.program ⟨257⟩, ⟨32278⟩⟩
def transferEvent : Nat := 41906
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41904 .coefficient, .predecessor 1 41905 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41904 .coefficient)
      LeftBound41902.bound (LeftBound41902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41905 .coefficient)
      LeftAuthority41825.bound (LeftAuthority41825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41825.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41825.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41902.bound, LeftAuthority41825.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41902.bound, LeftAuthority41825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41902.actual selector witness, LeftAuthority41825.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41906

namespace LeftBound41910
def owner : Owner := ⟨.program ⟨257⟩, ⟨51333⟩⟩
def transferEvent : Nat := 41910
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41908 .coefficient, .predecessor 1 41909 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41908 .coefficient)
      LeftBound41906.bound (LeftBound41906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41909 .coefficient)
      LeftAuthority41802.bound (LeftAuthority41802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41802.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41906.bound, LeftAuthority41802.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41906.bound, LeftAuthority41802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41906.actual selector witness, LeftAuthority41802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41910

namespace LeftBound41914
def owner : Owner := ⟨.program ⟨257⟩, ⟨54313⟩⟩
def transferEvent : Nat := 41914
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41912 .coefficient, .predecessor 1 41913 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41912 .coefficient)
      LeftBound41910.bound (LeftBound41910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41913 .coefficient)
      LeftAuthority41779.bound (LeftAuthority41779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41779.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41910.bound, LeftAuthority41779.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41910.bound, LeftAuthority41779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41910.actual selector witness, LeftAuthority41779.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41914

namespace LeftBound41918
def owner : Owner := ⟨.program ⟨257⟩, ⟨57293⟩⟩
def transferEvent : Nat := 41918
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41916 .coefficient, .predecessor 1 41917 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41916 .coefficient)
      LeftBound41914.bound (LeftBound41914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41917 .coefficient)
      LeftAuthority41756.bound (LeftAuthority41756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41756.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41914.bound, LeftAuthority41756.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41914.bound, LeftAuthority41756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41914.actual selector witness, LeftAuthority41756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41918

namespace LeftBound41922
def owner : Owner := ⟨.program ⟨257⟩, ⟨60273⟩⟩
def transferEvent : Nat := 41922
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41920 .coefficient, .predecessor 1 41921 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41920 .coefficient)
      LeftBound41918.bound (LeftBound41918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41918.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41921 .coefficient)
      LeftAuthority41733.bound (LeftAuthority41733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41733.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41733.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41918.bound, LeftAuthority41733.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41918.bound, LeftAuthority41733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41918.actual selector witness, LeftAuthority41733.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41922

namespace LeftBound41926
def owner : Owner := ⟨.program ⟨257⟩, ⟨63253⟩⟩
def transferEvent : Nat := 41926
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41924 .coefficient, .predecessor 1 41925 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41924 .coefficient)
      LeftBound41922.bound (LeftBound41922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41925 .coefficient)
      LeftAuthority41710.bound (LeftAuthority41710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41710.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41922.bound, LeftAuthority41710.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41922.bound, LeftAuthority41710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41922.actual selector witness, LeftAuthority41710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41926

namespace LeftBound41930
def owner : Owner := ⟨.program ⟨257⟩, ⟨67232⟩⟩
def transferEvent : Nat := 41930
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41928 .coefficient, .predecessor 1 41929 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41928 .coefficient)
      LeftBound41926.bound (LeftBound41926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41929 .coefficient)
      LeftAuthority41687.bound (LeftAuthority41687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41687.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41926.bound, LeftAuthority41687.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41926.bound, LeftAuthority41687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41926.actual selector witness, LeftAuthority41687.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41930

namespace LeftBound41934
def owner : Owner := ⟨.program ⟨257⟩, ⟨67233⟩⟩
def transferEvent : Nat := 41934
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41932 .coefficient, .predecessor 1 41933 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41932 .coefficient)
      LeftBound41930.bound (LeftBound41930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41933 .coefficient)
      LeftAuthority41664.bound (LeftAuthority41664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41930.bound, LeftAuthority41664.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41930.bound, LeftAuthority41664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41930.actual selector witness, LeftAuthority41664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41934

namespace LeftBound41938
def owner : Owner := ⟨.program ⟨257⟩, ⟨67234⟩⟩
def transferEvent : Nat := 41938
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41936 .coefficient, .predecessor 1 41937 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41936 .coefficient)
      LeftBound41934.bound (LeftBound41934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41937 .coefficient)
      LeftAuthority41641.bound (LeftAuthority41641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41641.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41934.bound, LeftAuthority41641.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41934.bound, LeftAuthority41641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41934.actual selector witness, LeftAuthority41641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41938

namespace LeftBound41942
def owner : Owner := ⟨.program ⟨257⟩, ⟨67235⟩⟩
def transferEvent : Nat := 41942
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41940 .coefficient, .predecessor 1 41941 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41940 .coefficient)
      LeftBound41938.bound (LeftBound41938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41941 .coefficient)
      LeftAuthority41618.bound (LeftAuthority41618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41618.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41938.bound, LeftAuthority41618.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41938.bound, LeftAuthority41618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41938.actual selector witness, LeftAuthority41618.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41942

namespace LeftBound41946
def owner : Owner := ⟨.program ⟨257⟩, ⟨67236⟩⟩
def transferEvent : Nat := 41946
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41944 .coefficient, .predecessor 1 41945 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41944 .coefficient)
      LeftBound41942.bound (LeftBound41942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41945 .coefficient)
      LeftAuthority41595.bound (LeftAuthority41595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41595.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41942.bound, LeftAuthority41595.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41942.bound, LeftAuthority41595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41942.actual selector witness, LeftAuthority41595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41946

namespace LeftBound41950
def owner : Owner := ⟨.program ⟨257⟩, ⟨67237⟩⟩
def transferEvent : Nat := 41950
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41948 .coefficient, .predecessor 1 41949 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41948 .coefficient)
      LeftBound41946.bound (LeftBound41946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41949 .coefficient)
      LeftAuthority41572.bound (LeftAuthority41572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41572.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41946.bound, LeftAuthority41572.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41946.bound, LeftAuthority41572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41946.actual selector witness, LeftAuthority41572.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41950

namespace LeftBound41954
def owner : Owner := ⟨.program ⟨257⟩, ⟨67238⟩⟩
def transferEvent : Nat := 41954
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41952 .coefficient, .predecessor 1 41953 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41952 .coefficient)
      LeftBound41950.bound (LeftBound41950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41950.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41953 .coefficient)
      LeftAuthority41549.bound (LeftAuthority41549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41950.bound, LeftAuthority41549.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41950.bound, LeftAuthority41549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41950.actual selector witness, LeftAuthority41549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41954

namespace LeftBound41958
def owner : Owner := ⟨.program ⟨257⟩, ⟨67239⟩⟩
def transferEvent : Nat := 41958
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41956 .coefficient, .predecessor 1 41957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41956 .coefficient)
      LeftBound41954.bound (LeftBound41954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41957 .coefficient)
      LeftAuthority41526.bound (LeftAuthority41526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41526.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41954.bound, LeftAuthority41526.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41954.bound, LeftAuthority41526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41954.actual selector witness, LeftAuthority41526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41958

namespace LeftBound41962
def owner : Owner := ⟨.program ⟨257⟩, ⟨67240⟩⟩
def transferEvent : Nat := 41962
def frameStart : Nat := 41461
def rule : BoundRule := .sum [.predecessor 0 41960 .coefficient, .predecessor 1 41961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 41960 .coefficient)
      LeftBound41958.bound (LeftBound41958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 41961 .coefficient)
      LeftAuthority41503.bound (LeftAuthority41503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41503.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41958.bound, LeftAuthority41503.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41958.bound, LeftAuthority41503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound41958.actual selector witness, LeftAuthority41503.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41962

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
