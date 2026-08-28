import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard126
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard171
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard173
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard227

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound39197
def owner : Owner := ⟨.program ⟨257⟩, ⟨33344⟩⟩
def transferEvent : Nat := 39197
def frameStart : Nat := 39132
def rule : BoundRule := .product (.predecessor 0 39195 .coefficient) (.predecessor 1 39196 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39195 .coefficient)
      LeftAuthority39193.bound (LeftAuthority39193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39193.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39196 .coefficient)
      LeftBound39191.bound (LeftBound39191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39191.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority39193.bound LeftBound39191.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39193.bound, LeftBound39191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority39193.actual selector witness) * (LeftBound39191.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39197

namespace LeftBound39205
def owner : Owner := ⟨.program ⟨257⟩, ⟨33345⟩⟩
def transferEvent : Nat := 39205
def frameStart : Nat := 39132
def rule : BoundRule := .sum [.predecessor 0 39203 .coefficient, .predecessor 1 39204 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39203 .coefficient)
      LeftAuthority39201.bound (LeftAuthority39201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39201.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39204 .coefficient)
      LeftBound39197.bound (LeftBound39197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39197.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority39201.bound, LeftBound39197.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39201.bound, LeftBound39197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority39201.actual selector witness, LeftBound39197.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39205

namespace LeftBound39209
def owner : Owner := ⟨.program ⟨257⟩, ⟨34172⟩⟩
def transferEvent : Nat := 39209
def frameStart : Nat := 39132
def rule : BoundRule := .product (.predecessor 0 39207 .coefficient) (.predecessor 1 39208 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39207 .coefficient)
      LeftBound39205.bound (LeftBound39205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39205.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39208 .coefficient)
      LeftAuthority39182.bound (LeftAuthority39182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound39205.bound LeftAuthority39182.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39205.bound, LeftAuthority39182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound39205.actual selector witness) * (LeftAuthority39182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39209

namespace LeftBound39220
def owner : Owner := ⟨.program ⟨257⟩, ⟨32279⟩⟩
def transferEvent : Nat := 39220
def frameStart : Nat := 39132
def rule : BoundRule := .product (.predecessor 0 39218 .coefficient) (.predecessor 1 39219 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39218 .coefficient)
      LeftAuthority39193.bound (LeftAuthority39193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39193.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39219 .coefficient)
      LeftAuthority39216.bound (LeftAuthority39216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39216.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39216.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority39193.bound LeftAuthority39216.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39193.bound, LeftAuthority39216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority39193.actual selector witness) * (LeftAuthority39216.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39220

namespace LeftBound39228
def owner : Owner := ⟨.program ⟨257⟩, ⟨32280⟩⟩
def transferEvent : Nat := 39228
def frameStart : Nat := 39132
def rule : BoundRule := .sum [.predecessor 0 39226 .coefficient, .predecessor 1 39227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39226 .coefficient)
      LeftAuthority39224.bound (LeftAuthority39224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39224.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39227 .coefficient)
      LeftBound39220.bound (LeftBound39220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority39224.bound, LeftBound39220.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39224.bound, LeftBound39220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority39224.actual selector witness, LeftBound39220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39228

namespace LeftBound39232
def owner : Owner := ⟨.program ⟨257⟩, ⟨34176⟩⟩
def transferEvent : Nat := 39232
def frameStart : Nat := 39132
def rule : BoundRule := .sum [.predecessor 0 39230 .coefficient, .predecessor 1 39231 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39230 .coefficient)
      LeftBound39228.bound (LeftBound39228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39231 .coefficient)
      LeftBound39209.bound (LeftBound39209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39209.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39228.bound, LeftBound39209.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39228.bound, LeftBound39209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound39228.actual selector witness, LeftBound39209.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39232

namespace LeftBound39245
def owner : Owner := ⟨.program ⟨257⟩, ⟨34174⟩⟩
def transferEvent : Nat := 39245
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39243 .coefficient, .predecessor 1 39244 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39243 .coefficient)
      LeftBound39074.bound (LeftBound39074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39244 .coefficient)
      LeftBound39057.bound (LeftBound39057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39057.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39074.bound, LeftBound39057.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39074.bound, LeftBound39057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound39074.actual selector witness, LeftBound39057.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39245

namespace LeftBound39248
def owner : Owner := ⟨.program ⟨257⟩, ⟨34174⟩⟩
def transferEvent : Nat := 39248
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 39242 .summary, .result 39064 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 39242 .summary)
      LeftBound39076.bound (LeftBound39076.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32879⟩⟩) (rawTerms := some (Proof.Events153.exact39242RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 39064 .summary)
      LeftBound39059.bound (LeftBound39059.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34173⟩⟩) (rawTerms := some (Proof.Events152.exact39064RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39076.bound, LeftBound39059.bound]
def bound : CoeffClass := .finite ⟨32189200113375081643992404983808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39076.bound, LeftBound39059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound39076.actual selector witness, LeftBound39059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39248

namespace LeftBound39272
def owner : Owner := ⟨.program ⟨257⟩, ⟨21713⟩⟩
def transferEvent : Nat := 39272
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 39270 .coefficient) (.predecessor 1 39271 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39270 .coefficient)
      LeftAuthority1186.bound (LeftAuthority1186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39271 .coefficient)
      LeftBound32026.bound (LeftBound32026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority1186.bound LeftBound32026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1186.bound, LeftBound32026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority1186.actual selector witness) * (LeftBound32026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39272

namespace LeftBound39277
def owner : Owner := ⟨.program ⟨257⟩, ⟨11639⟩⟩
def transferEvent : Nat := 39277
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39275 .coefficient) (.predecessor 1 39276 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39275 .coefficient)
      LeftBound31897.bound (LeftBound31897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39276 .coefficient)
      LeftBound24594.bound (LeftBound24594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24594.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound31897.bound LeftBound24594.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31897.bound, LeftBound24594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound31897.actual selector witness) * (LeftBound24594.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39277

namespace LeftBound39282
def owner : Owner := ⟨.program ⟨257⟩, ⟨21714⟩⟩
def transferEvent : Nat := 39282
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39280 .coefficient, .predecessor 1 39281 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39280 .coefficient)
      LeftBound39277.bound (LeftBound39277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39277.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39281 .coefficient)
      LeftBound39272.bound (LeftBound39272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39272.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39277.bound, LeftBound39272.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39277.bound, LeftBound39272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound39277.actual selector witness, LeftBound39272.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39282

namespace LeftBound39286
def owner : Owner := ⟨.program ⟨257⟩, ⟨21715⟩⟩
def transferEvent : Nat := 39286
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39284 .coefficient, .predecessor 1 39285 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39284 .coefficient)
      LeftBound39282.bound (LeftBound39282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39285 .coefficient)
      LeftBound24586.bound (LeftBound24586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24586.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39282.bound, LeftBound24586.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39282.bound, LeftBound24586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound39282.actual selector witness, LeftBound24586.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39286

namespace LeftBound39287
def owner : Owner := ⟨.program ⟨257⟩, ⟨21715⟩⟩
def transferEvent : Nat := 39287
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨132⟩⟩]⟩ [⟨.result 24587 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24587 .coefficient)
      LeftBound24586.bound (LeftBound24586.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨132⟩⟩) (rawTerms := some (Proof.Events096.exact24587RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24586.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound24586.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound24586.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39287

namespace LeftBound39292
def owner : Owner := ⟨.program ⟨257⟩, ⟨21716⟩⟩
def transferEvent : Nat := 39292
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39290 .coefficient) (.predecessor 1 39291 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39290 .coefficient)
      LeftBound39286.bound (LeftBound39286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39291 .coefficient)
      LeftAuthority1189.bound (LeftAuthority1189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1189.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1189.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound39286.bound LeftAuthority1189.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39286.bound, LeftAuthority1189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound39286.actual selector witness) * (LeftAuthority1189.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39292

namespace LeftBound39293
def owner : Owner := ⟨.program ⟨257⟩, ⟨21716⟩⟩
def transferEvent : Nat := 39293
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨21236⟩⟩], []⟩ [⟨.result 1190 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 1190 .coefficient)
      LeftAuthority1189.bound (LeftAuthority1189.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨21236⟩⟩) (rawTerms := some (Proof.Events004.exact1190RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1189.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1189.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1189.bound []
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority1189.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39293

namespace LeftBound39294
def owner : Owner := ⟨.program ⟨257⟩, ⟨21716⟩⟩
def transferEvent : Nat := 39294
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 39289 .summary) (.transfer 39293) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 39289 .summary)
      LeftBound39287.bound (LeftBound39287.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨21715⟩⟩) (rawTerms := some (Proof.Events153.exact39289RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 39293)
      LeftBound39293.bound (LeftBound39293.actual selector witness) := by
  exact .transfer (LeftBound39293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound39287.bound LeftBound39293.bound
def bound : CoeffClass := .finite ⟨3407872, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39287.bound, LeftBound39293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound39287.actual selector witness) * (LeftBound39293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39294

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
