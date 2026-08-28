import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard133
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard134
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2057

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound302394
def owner : Owner := ⟨.program ⟨257⟩, ⟨18509⟩⟩
def transferEvent : Nat := 302394
def frameStart : Nat := 302367
def rule : BoundRule := .identity (.predecessor 0 302393 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302393 .coefficient)
      LeftAuthority302391.bound (LeftAuthority302391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302391.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302391.derived selector witness)

def rawBound : CoeffClass := LeftAuthority302391.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority302391.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound302394

namespace LeftBound302411
def owner : Owner := ⟨.program ⟨257⟩, ⟨20026⟩⟩
def transferEvent : Nat := 302411
def frameStart : Nat := 302367
def rule : BoundRule := .sum [.predecessor 0 302409 .coefficient, .predecessor 1 302410 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302409 .coefficient)
      LeftBound302394.bound (LeftBound302394.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound302394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302410 .coefficient)
      LeftAuthority302407.bound (LeftAuthority302407.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority302407.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302394.bound, LeftAuthority302407.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302394.bound, LeftAuthority302407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302394.actual selector witness, LeftAuthority302407.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302411

namespace LeftBound302414
def owner : Owner := ⟨.program ⟨257⟩, ⟨20027⟩⟩
def transferEvent : Nat := 302414
def frameStart : Nat := 302367
def rule : BoundRule := .identity (.predecessor 0 302413 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302413 .coefficient)
      LeftBound302411.bound (LeftBound302411.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound302411.derived selector witness)

def rawBound : CoeffClass := LeftBound302411.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound302411.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound302414

namespace LeftBound302420
def owner : Owner := ⟨.program ⟨257⟩, ⟨20028⟩⟩
def transferEvent : Nat := 302420
def frameStart : Nat := 302367
def rule : BoundRule := .product (.predecessor 0 302418 .coefficient) (.predecessor 1 302419 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302418 .coefficient)
      LeftAuthority302416.bound (LeftAuthority302416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302419 .coefficient)
      LeftBound302414.bound (LeftBound302414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority302416.bound LeftBound302414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302416.bound, LeftBound302414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority302416.actual selector witness) * (LeftBound302414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302420

namespace LeftBound302428
def owner : Owner := ⟨.program ⟨257⟩, ⟨20029⟩⟩
def transferEvent : Nat := 302428
def frameStart : Nat := 302367
def rule : BoundRule := .sum [.predecessor 0 302426 .coefficient, .predecessor 1 302427 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302426 .coefficient)
      LeftAuthority302424.bound (LeftAuthority302424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302424.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302427 .coefficient)
      LeftBound302420.bound (LeftBound302420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302420.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302420.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority302424.bound, LeftBound302420.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302424.bound, LeftBound302420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority302424.actual selector witness, LeftBound302420.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302428

namespace LeftBound302432
def owner : Owner := ⟨.program ⟨257⟩, ⟨20343⟩⟩
def transferEvent : Nat := 302432
def frameStart : Nat := 302367
def rule : BoundRule := .product (.predecessor 0 302430 .coefficient) (.predecessor 1 302431 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302430 .coefficient)
      LeftBound302428.bound (LeftBound302428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302428.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302428.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302431 .coefficient)
      LeftAuthority302405.bound (LeftAuthority302405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302405.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302428.bound LeftAuthority302405.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302428.bound, LeftAuthority302405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302428.actual selector witness) * (LeftAuthority302405.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302432

namespace LeftBound302443
def owner : Owner := ⟨.program ⟨257⟩, ⟨18678⟩⟩
def transferEvent : Nat := 302443
def frameStart : Nat := 302367
def rule : BoundRule := .product (.predecessor 0 302441 .coefficient) (.predecessor 1 302442 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302441 .coefficient)
      LeftAuthority302416.bound (LeftAuthority302416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302442 .coefficient)
      LeftAuthority302439.bound (LeftAuthority302439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302439.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302439.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority302416.bound LeftAuthority302439.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302416.bound, LeftAuthority302439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority302416.actual selector witness) * (LeftAuthority302439.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302443

namespace LeftBound302451
def owner : Owner := ⟨.program ⟨257⟩, ⟨18679⟩⟩
def transferEvent : Nat := 302451
def frameStart : Nat := 302367
def rule : BoundRule := .sum [.predecessor 0 302449 .coefficient, .predecessor 1 302450 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302449 .coefficient)
      LeftAuthority302447.bound (LeftAuthority302447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302450 .coefficient)
      LeftBound302443.bound (LeftBound302443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority302447.bound, LeftBound302443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302447.bound, LeftBound302443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority302447.actual selector witness, LeftBound302443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302451

namespace LeftBound302455
def owner : Owner := ⟨.program ⟨257⟩, ⟨20347⟩⟩
def transferEvent : Nat := 302455
def frameStart : Nat := 302367
def rule : BoundRule := .sum [.predecessor 0 302453 .coefficient, .predecessor 1 302454 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302453 .coefficient)
      LeftBound302451.bound (LeftBound302451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302451.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302454 .coefficient)
      LeftBound302432.bound (LeftBound302432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302451.bound, LeftBound302432.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302451.bound, LeftBound302432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302451.actual selector witness, LeftBound302432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302455

namespace LeftBound302468
def owner : Owner := ⟨.program ⟨257⟩, ⟨20345⟩⟩
def transferEvent : Nat := 302468
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302466 .coefficient, .predecessor 1 302467 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302466 .coefficient)
      LeftBound302321.bound (LeftBound302321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302467 .coefficient)
      LeftBound302304.bound (LeftBound302304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302304.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302304.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302321.bound, LeftBound302304.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302321.bound, LeftBound302304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302321.actual selector witness, LeftBound302304.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302468

namespace LeftBound302471
def owner : Owner := ⟨.program ⟨257⟩, ⟨20345⟩⟩
def transferEvent : Nat := 302471
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302465 .summary, .result 302311 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302465 .summary)
      LeftBound302323.bound (LeftBound302323.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19259⟩⟩) (rawTerms := some (Proof.Events1181.exact302465RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302311 .summary)
      LeftBound302306.bound (LeftBound302306.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20344⟩⟩) (rawTerms := some (Proof.Events1180.exact302311RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302306.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302323.bound, LeftBound302306.bound]
def bound : CoeffClass := .finite ⟨32188905437706550578131070353408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302323.bound, LeftBound302306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302323.actual selector witness, LeftBound302306.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302471

namespace LeftBound302495
def owner : Owner := ⟨.program ⟨257⟩, ⟨15237⟩⟩
def transferEvent : Nat := 302495
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 302493 .coefficient) (.predecessor 1 302494 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302493 .coefficient)
      LeftAuthority14680.bound (LeftAuthority14680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14680.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302494 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority14680.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14680.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority14680.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound302495

namespace LeftBound302500
def owner : Owner := ⟨.program ⟨257⟩, ⟨7452⟩⟩
def transferEvent : Nat := 302500
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 302498 .coefficient) (.predecessor 1 302499 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302498 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302499 .coefficient)
      LeftBound25596.bound (LeftBound25596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25596.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftBound25596.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound25596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftBound25596.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302500

namespace LeftBound302505
def owner : Owner := ⟨.program ⟨257⟩, ⟨15238⟩⟩
def transferEvent : Nat := 302505
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302503 .coefficient, .predecessor 1 302504 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302503 .coefficient)
      LeftBound302500.bound (LeftBound302500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302504 .coefficient)
      LeftBound302495.bound (LeftBound302495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302495.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302500.bound, LeftBound302495.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302500.bound, LeftBound302495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302500.actual selector witness, LeftBound302495.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302505

namespace LeftBound302509
def owner : Owner := ⟨.program ⟨257⟩, ⟨15239⟩⟩
def transferEvent : Nat := 302509
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302507 .coefficient, .predecessor 1 302508 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302507 .coefficient)
      LeftBound302505.bound (LeftBound302505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1181.exact302506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302508 .coefficient)
      LeftBound25588.bound (LeftBound25588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25588.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302505.bound, LeftBound25588.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302505.bound, LeftBound25588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302505.actual selector witness, LeftBound25588.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302509

namespace LeftBound302510
def owner : Owner := ⟨.program ⟨257⟩, ⟨15239⟩⟩
def transferEvent : Nat := 302510
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨130⟩⟩]⟩ [⟨.result 25589 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25589 .coefficient)
      LeftBound25588.bound (LeftBound25588.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨130⟩⟩) (rawTerms := some (Proof.Events099.exact25589RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25588.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25588.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25588.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound302510

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
