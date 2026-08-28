import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard086
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1388
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1391
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1408

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound209877
def owner : Owner := ⟨.program ⟨257⟩, ⟨38788⟩⟩
def transferEvent : Nat := 209877
def frameStart : Nat := 209812
def rule : BoundRule := .product (.predecessor 0 209875 .coefficient) (.predecessor 1 209876 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209875 .coefficient)
      LeftAuthority209873.bound (LeftAuthority209873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority209873.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority209873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209876 .coefficient)
      LeftBound209871.bound (LeftBound209871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209871.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority209873.bound LeftBound209871.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority209873.bound, LeftBound209871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority209873.actual selector witness) * (LeftBound209871.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound209877

namespace LeftBound209885
def owner : Owner := ⟨.program ⟨257⟩, ⟨38789⟩⟩
def transferEvent : Nat := 209885
def frameStart : Nat := 209812
def rule : BoundRule := .sum [.predecessor 0 209883 .coefficient, .predecessor 1 209884 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209883 .coefficient)
      LeftAuthority209881.bound (LeftAuthority209881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority209881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority209881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209884 .coefficient)
      LeftBound209877.bound (LeftBound209877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209877.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209877.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority209881.bound, LeftBound209877.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority209881.bound, LeftBound209877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority209881.actual selector witness, LeftBound209877.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209885

namespace LeftBound209889
def owner : Owner := ⟨.program ⟨257⟩, ⟨39310⟩⟩
def transferEvent : Nat := 209889
def frameStart : Nat := 209812
def rule : BoundRule := .product (.predecessor 0 209887 .coefficient) (.predecessor 1 209888 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209887 .coefficient)
      LeftBound209885.bound (LeftBound209885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209885.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209885.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209888 .coefficient)
      LeftAuthority209862.bound (LeftAuthority209862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority209862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority209862.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound209885.bound LeftAuthority209862.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209885.bound, LeftAuthority209862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound209885.actual selector witness) * (LeftAuthority209862.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound209889

namespace LeftBound209900
def owner : Owner := ⟨.program ⟨257⟩, ⟨37644⟩⟩
def transferEvent : Nat := 209900
def frameStart : Nat := 209812
def rule : BoundRule := .product (.predecessor 0 209898 .coefficient) (.predecessor 1 209899 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209898 .coefficient)
      LeftAuthority209873.bound (LeftAuthority209873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority209873.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority209873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209899 .coefficient)
      LeftAuthority209896.bound (LeftAuthority209896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority209896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority209896.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority209873.bound LeftAuthority209896.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority209873.bound, LeftAuthority209896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority209873.actual selector witness) * (LeftAuthority209896.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound209900

namespace LeftBound209908
def owner : Owner := ⟨.program ⟨257⟩, ⟨37645⟩⟩
def transferEvent : Nat := 209908
def frameStart : Nat := 209812
def rule : BoundRule := .sum [.predecessor 0 209906 .coefficient, .predecessor 1 209907 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209906 .coefficient)
      LeftAuthority209904.bound (LeftAuthority209904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority209904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority209904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209907 .coefficient)
      LeftBound209900.bound (LeftBound209900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209900.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority209904.bound, LeftBound209900.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority209904.bound, LeftBound209900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority209904.actual selector witness, LeftBound209900.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209908

namespace LeftBound209912
def owner : Owner := ⟨.program ⟨257⟩, ⟨39313⟩⟩
def transferEvent : Nat := 209912
def frameStart : Nat := 209812
def rule : BoundRule := .sum [.predecessor 0 209910 .coefficient, .predecessor 1 209911 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209910 .coefficient)
      LeftBound209908.bound (LeftBound209908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209911 .coefficient)
      LeftBound209889.bound (LeftBound209889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound209908.bound, LeftBound209889.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209908.bound, LeftBound209889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound209908.actual selector witness, LeftBound209889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209912

namespace LeftBound209925
def owner : Owner := ⟨.program ⟨257⟩, ⟨39312⟩⟩
def transferEvent : Nat := 209925
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 209923 .coefficient, .predecessor 1 209924 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209923 .coefficient)
      LeftBound209754.bound (LeftBound209754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events820.exact209922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209924 .coefficient)
      LeftBound209737.bound (LeftBound209737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events819.exact209744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209737.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209737.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound209754.bound, LeftBound209737.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209754.bound, LeftBound209737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound209754.actual selector witness, LeftBound209737.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209925

namespace LeftBound209928
def owner : Owner := ⟨.program ⟨257⟩, ⟨39312⟩⟩
def transferEvent : Nat := 209928
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 209922 .summary, .result 209744 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 209922 .summary)
      LeftBound209756.bound (LeftBound209756.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38179⟩⟩) (rawTerms := some (Proof.Events820.exact209922RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound209756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 209744 .summary)
      LeftBound209739.bound (LeftBound209739.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39311⟩⟩) (rawTerms := some (Proof.Events819.exact209744RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound209739.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound209756.bound, LeftBound209739.bound]
def bound : CoeffClass := .finite ⟨32192736221397454434328420548608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209756.bound, LeftBound209739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound209756.actual selector witness, LeftBound209739.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209928

namespace LeftBound209952
def owner : Owner := ⟨.program ⟨257⟩, ⟨34437⟩⟩
def transferEvent : Nat := 209952
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 209950 .coefficient) (.predecessor 1 209951 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209950 .coefficient)
      LeftAuthority9932.bound (LeftAuthority9932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209951 .coefficient)
      LeftBound207526.bound (LeftBound207526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority9932.bound LeftBound207526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9932.bound, LeftBound207526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority9932.actual selector witness) * (LeftBound207526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound209952

namespace LeftBound209957
def owner : Owner := ⟨.program ⟨257⟩, ⟨8586⟩⟩
def transferEvent : Nat := 209957
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 209955 .coefficient) (.predecessor 1 209956 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209955 .coefficient)
      LeftBound207397.bound (LeftBound207397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209956 .coefficient)
      LeftBound19584.bound (LeftBound19584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19584.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound207397.bound LeftBound19584.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207397.bound, LeftBound19584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound207397.actual selector witness) * (LeftBound19584.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound209957

namespace LeftBound209962
def owner : Owner := ⟨.program ⟨257⟩, ⟨34438⟩⟩
def transferEvent : Nat := 209962
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 209960 .coefficient, .predecessor 1 209961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209960 .coefficient)
      LeftBound209957.bound (LeftBound209957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events820.exact209959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209957.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209961 .coefficient)
      LeftBound209952.bound (LeftBound209952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events820.exact209954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209952.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound209957.bound, LeftBound209952.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209957.bound, LeftBound209952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound209957.actual selector witness, LeftBound209952.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209962

namespace LeftBound209966
def owner : Owner := ⟨.program ⟨257⟩, ⟨34439⟩⟩
def transferEvent : Nat := 209966
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 209964 .coefficient, .predecessor 1 209965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209964 .coefficient)
      LeftBound209962.bound (LeftBound209962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events820.exact209963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209965 .coefficient)
      LeftBound19576.bound (LeftBound19576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound209962.bound, LeftBound19576.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209962.bound, LeftBound19576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound209962.actual selector witness, LeftBound19576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209966

namespace LeftBound209967
def owner : Owner := ⟨.program ⟨257⟩, ⟨34439⟩⟩
def transferEvent : Nat := 209967
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨106⟩⟩]⟩ [⟨.result 19577 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19577 .coefficient)
      LeftBound19576.bound (LeftBound19576.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨106⟩⟩) (rawTerms := some (Proof.Events076.exact19577RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19576.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19576.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19576.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound209967

namespace LeftBound209972
def owner : Owner := ⟨.program ⟨257⟩, ⟨34440⟩⟩
def transferEvent : Nat := 209972
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 209970 .coefficient) (.predecessor 1 209971 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209970 .coefficient)
      LeftBound209966.bound (LeftBound209966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events820.exact209969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209971 .coefficient)
      LeftAuthority9935.bound (LeftAuthority9935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9935.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9935.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound209966.bound LeftAuthority9935.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209966.bound, LeftAuthority9935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound209966.actual selector witness) * (LeftAuthority9935.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound209972

namespace LeftBound209973
def owner : Owner := ⟨.program ⟨257⟩, ⟨34440⟩⟩
def transferEvent : Nat := 209973
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13581⟩⟩], []⟩ [⟨.result 9936 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 9936 .coefficient)
      LeftAuthority9935.bound (LeftAuthority9935.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13581⟩⟩) (rawTerms := some (Proof.Events038.exact9936RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9935.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9935.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9935.bound []
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority9935.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound209973

namespace LeftBound209974
def owner : Owner := ⟨.program ⟨257⟩, ⟨34440⟩⟩
def transferEvent : Nat := 209974
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 209969 .summary) (.transfer 209973) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 209969 .summary)
      LeftBound209967.bound (LeftBound209967.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34439⟩⟩) (rawTerms := some (Proof.Events820.exact209969RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound209967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 209973)
      LeftBound209973.bound (LeftBound209973.actual selector witness) := by
  exact .transfer (LeftBound209973.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound209967.bound LeftBound209973.bound
def bound : CoeffClass := .finite ⟨34078720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209967.bound, LeftBound209973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound209967.actual selector witness) * (LeftBound209973.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound209974

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
