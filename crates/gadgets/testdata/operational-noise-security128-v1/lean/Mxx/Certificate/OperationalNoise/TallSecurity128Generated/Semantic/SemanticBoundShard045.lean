import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard001
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard044

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound14160
def owner : Owner := ⟨.program ⟨257⟩, ⟨31988⟩⟩
def transferEvent : Nat := 14160
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14158 .coefficient) (.predecessor 1 14159 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14158 .coefficient)
      LeftAuthority14156.bound (LeftAuthority14156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14156.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14159 .coefficient)
      LeftAuthority682.bound (LeftAuthority682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14156.bound LeftAuthority682.bound
def bound : CoeffClass := .finite ⟨201065796616126235971320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14156.bound, LeftAuthority682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14156.actual selector witness) * (LeftAuthority682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14160

namespace LeftBound14168
def owner : Owner := ⟨.program ⟨257⟩, ⟨21968⟩⟩
def transferEvent : Nat := 14168
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14166 .coefficient) (.predecessor 1 14167 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14166 .coefficient)
      LeftAuthority14164.bound (LeftAuthority14164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14167 .coefficient)
      LeftAuthority692.bound (LeftAuthority692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority692.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14164.bound LeftAuthority692.bound
def bound : CoeffClass := .finite ⟨187661410175051153573232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14164.bound, LeftAuthority692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14164.actual selector witness) * (LeftAuthority692.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14168

namespace LeftBound14176
def owner : Owner := ⟨.program ⟨257⟩, ⟨18748⟩⟩
def transferEvent : Nat := 14176
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14174 .coefficient) (.predecessor 1 14175 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14174 .coefficient)
      LeftAuthority14172.bound (LeftAuthority14172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14172.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14175 .coefficient)
      LeftAuthority702.bound (LeftAuthority702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority702.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14172.bound LeftAuthority702.bound
def bound : CoeffClass := .finite ⟨175932572039110456474905, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14172.bound, LeftAuthority702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14172.actual selector witness) * (LeftAuthority702.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14176

namespace LeftBound14184
def owner : Owner := ⟨.program ⟨257⟩, ⟨15935⟩⟩
def transferEvent : Nat := 14184
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14182 .coefficient) (.predecessor 1 14183 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14182 .coefficient)
      LeftAuthority14180.bound (LeftAuthority14180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14183 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14180.bound LeftAuthority712.bound
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14180.bound, LeftAuthority712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14180.actual selector witness) * (LeftAuthority712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14184

namespace LeftBound14189
def owner : Owner := ⟨.program ⟨257⟩, ⟨15936⟩⟩
def transferEvent : Nat := 14189
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14187 .coefficient, .predecessor 1 14188 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14187 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14188 .coefficient)
      LeftBound14184.bound (LeftBound14184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14184.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound14184.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644362, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound14184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound726.actual selector witness, LeftBound14184.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14189

namespace LeftBound14193
def owner : Owner := ⟨.program ⟨257⟩, ⟨18749⟩⟩
def transferEvent : Nat := 14193
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14191 .coefficient, .predecessor 1 14192 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14191 .coefficient)
      LeftBound14189.bound (LeftBound14189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14192 .coefficient)
      LeftBound14176.bound (LeftBound14176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14189.bound, LeftBound14176.bound]
def bound : CoeffClass := .finite ⟨332317080518319751119267, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14189.bound, LeftBound14176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14189.actual selector witness, LeftBound14176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14193

namespace LeftBound14197
def owner : Owner := ⟨.program ⟨257⟩, ⟨21969⟩⟩
def transferEvent : Nat := 14197
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14195 .coefficient, .predecessor 1 14196 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14195 .coefficient)
      LeftBound14193.bound (LeftBound14193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14196 .coefficient)
      LeftBound14168.bound (LeftBound14168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14168.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14168.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14193.bound, LeftBound14168.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14193.bound, LeftBound14168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14193.actual selector witness, LeftBound14168.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14197

namespace LeftBound14201
def owner : Owner := ⟨.program ⟨257⟩, ⟨31989⟩⟩
def transferEvent : Nat := 14201
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14199 .coefficient, .predecessor 1 14200 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14199 .coefficient)
      LeftBound14197.bound (LeftBound14197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14200 .coefficient)
      LeftBound14160.bound (LeftBound14160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14160.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14197.bound, LeftBound14160.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14197.bound, LeftBound14160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14197.actual selector witness, LeftBound14160.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14201

namespace LeftBound14205
def owner : Owner := ⟨.program ⟨257⟩, ⟨51053⟩⟩
def transferEvent : Nat := 14205
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14203 .coefficient, .predecessor 1 14204 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14203 .coefficient)
      LeftBound14201.bound (LeftBound14201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14204 .coefficient)
      LeftBound14152.bound (LeftBound14152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14152.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14201.bound, LeftBound14152.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14201.bound, LeftBound14152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14201.actual selector witness, LeftBound14152.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14205

namespace LeftBound14209
def owner : Owner := ⟨.program ⟨257⟩, ⟨54033⟩⟩
def transferEvent : Nat := 14209
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14207 .coefficient, .predecessor 1 14208 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14207 .coefficient)
      LeftBound14205.bound (LeftBound14205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14205.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14208 .coefficient)
      LeftBound14144.bound (LeftBound14144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14144.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14144.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14205.bound, LeftBound14144.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14205.bound, LeftBound14144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14205.actual selector witness, LeftBound14144.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14209

namespace LeftBound14213
def owner : Owner := ⟨.program ⟨257⟩, ⟨57013⟩⟩
def transferEvent : Nat := 14213
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14211 .coefficient, .predecessor 1 14212 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14211 .coefficient)
      LeftBound14209.bound (LeftBound14209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14212 .coefficient)
      LeftBound14136.bound (LeftBound14136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14136.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14209.bound, LeftBound14136.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14209.bound, LeftBound14136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14209.actual selector witness, LeftBound14136.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14213

namespace LeftBound14217
def owner : Owner := ⟨.program ⟨257⟩, ⟨59993⟩⟩
def transferEvent : Nat := 14217
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14215 .coefficient, .predecessor 1 14216 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14215 .coefficient)
      LeftBound14213.bound (LeftBound14213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14216 .coefficient)
      LeftBound14128.bound (LeftBound14128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14213.bound, LeftBound14128.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14213.bound, LeftBound14128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14213.actual selector witness, LeftBound14128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14217

namespace LeftBound14221
def owner : Owner := ⟨.program ⟨257⟩, ⟨62973⟩⟩
def transferEvent : Nat := 14221
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14219 .coefficient, .predecessor 1 14220 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14219 .coefficient)
      LeftBound14217.bound (LeftBound14217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14217.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14220 .coefficient)
      LeftBound14120.bound (LeftBound14120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14120.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14217.bound, LeftBound14120.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14217.bound, LeftBound14120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14217.actual selector witness, LeftBound14120.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14221

namespace LeftBound14225
def owner : Owner := ⟨.program ⟨257⟩, ⟨66170⟩⟩
def transferEvent : Nat := 14225
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14223 .coefficient, .predecessor 1 14224 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14223 .coefficient)
      LeftBound14221.bound (LeftBound14221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14224 .coefficient)
      LeftBound14112.bound (LeftBound14112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14112.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14221.bound, LeftBound14112.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14221.bound, LeftBound14112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14221.actual selector witness, LeftBound14112.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14225

namespace LeftBound14229
def owner : Owner := ⟨.program ⟨257⟩, ⟨66171⟩⟩
def transferEvent : Nat := 14229
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14227 .coefficient, .predecessor 1 14228 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14227 .coefficient)
      LeftBound14225.bound (LeftBound14225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14225.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14228 .coefficient)
      LeftBound14104.bound (LeftBound14104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14104.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14225.bound, LeftBound14104.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14225.bound, LeftBound14104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14225.actual selector witness, LeftBound14104.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14229

namespace LeftBound14233
def owner : Owner := ⟨.program ⟨257⟩, ⟨66172⟩⟩
def transferEvent : Nat := 14233
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14231 .coefficient, .predecessor 1 14232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14231 .coefficient)
      LeftBound14229.bound (LeftBound14229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14232 .coefficient)
      LeftBound14096.bound (LeftBound14096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14096.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14229.bound, LeftBound14096.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14229.bound, LeftBound14096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14229.actual selector witness, LeftBound14096.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14233

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
