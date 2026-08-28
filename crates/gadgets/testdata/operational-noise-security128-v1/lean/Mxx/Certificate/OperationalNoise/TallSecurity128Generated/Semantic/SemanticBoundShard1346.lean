import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1345

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound200330
def owner : Owner := ⟨.program ⟨257⟩, ⟨23214⟩⟩
def transferEvent : Nat := 200330
def frameStart : Nat := 200280
def rule : BoundRule := .sum [.predecessor 0 200328 .coefficient, .predecessor 1 200329 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200328 .coefficient)
      LeftBound200313.bound (LeftBound200313.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound200313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200329 .coefficient)
      LeftAuthority200326.bound (LeftAuthority200326.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority200326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound200313.bound, LeftAuthority200326.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200313.bound, LeftAuthority200326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound200313.actual selector witness, LeftAuthority200326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200330

namespace LeftBound200333
def owner : Owner := ⟨.program ⟨257⟩, ⟨23215⟩⟩
def transferEvent : Nat := 200333
def frameStart : Nat := 200280
def rule : BoundRule := .identity (.predecessor 0 200332 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200332 .coefficient)
      LeftBound200330.bound (LeftBound200330.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound200330.derived selector witness)

def rawBound : CoeffClass := LeftBound200330.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound200330.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound200333

namespace LeftBound200339
def owner : Owner := ⟨.program ⟨257⟩, ⟨23216⟩⟩
def transferEvent : Nat := 200339
def frameStart : Nat := 200280
def rule : BoundRule := .product (.predecessor 0 200337 .coefficient) (.predecessor 1 200338 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200337 .coefficient)
      LeftAuthority200335.bound (LeftAuthority200335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200338 .coefficient)
      LeftBound200333.bound (LeftBound200333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200333.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority200335.bound LeftBound200333.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200335.bound, LeftBound200333.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority200335.actual selector witness) * (LeftBound200333.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200339

namespace LeftBound200355
def owner : Owner := ⟨.program ⟨257⟩, ⟨9575⟩⟩
def transferEvent : Nat := 200355
def frameStart : Nat := 200280
def rule : BoundRule := .scale (.predecessor 0 200353 .coefficient) (.value (.predecessor 1 200354 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200353 .coefficient)
      LeftAuthority200351.bound (LeftAuthority200351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200351.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200354 .coefficient)
      LeftAuthority200342.bound (LeftAuthority200342.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority200342.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority200351.bound LeftAuthority200342.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200351.bound, LeftAuthority200342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority200351.actual selector witness) * (LeftAuthority200342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound200355

namespace LeftBound200358
def owner : Owner := ⟨.program ⟨257⟩, ⟨7286⟩⟩
def transferEvent : Nat := 200358
def frameStart : Nat := 200280
def rule : BoundRule := .identity (.predecessor 0 200357 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200357 .coefficient)
      LeftAuthority200345.bound (LeftAuthority200345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200345.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200345.derived selector witness)

def rawBound : CoeffClass := LeftAuthority200345.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority200345.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound200358

namespace LeftBound200362
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def transferEvent : Nat := 200362
def frameStart : Nat := 200280
def rule : BoundRule := .product (.predecessor 0 200360 .coefficient) (.predecessor 1 200361 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200360 .coefficient)
      LeftBound200358.bound (LeftBound200358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200361 .coefficient)
      LeftBound200355.bound (LeftBound200355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200355.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound200358.bound LeftBound200355.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200358.bound, LeftBound200355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound200358.actual selector witness) * (LeftBound200355.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200362

namespace LeftBound200367
def owner : Owner := ⟨.program ⟨257⟩, ⟨23217⟩⟩
def transferEvent : Nat := 200367
def frameStart : Nat := 200280
def rule : BoundRule := .sum [.predecessor 0 200365 .coefficient, .predecessor 1 200366 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200365 .coefficient)
      LeftBound200362.bound (LeftBound200362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200366 .coefficient)
      LeftBound200339.bound (LeftBound200339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200339.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound200362.bound, LeftBound200339.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200362.bound, LeftBound200339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound200362.actual selector witness, LeftBound200339.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200367

namespace LeftBound200371
def owner : Owner := ⟨.program ⟨257⟩, ⟨23464⟩⟩
def transferEvent : Nat := 200371
def frameStart : Nat := 200280
def rule : BoundRule := .product (.predecessor 0 200369 .coefficient) (.predecessor 1 200370 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200369 .coefficient)
      LeftBound200367.bound (LeftBound200367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200370 .coefficient)
      LeftAuthority200324.bound (LeftAuthority200324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound200367.bound LeftAuthority200324.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200367.bound, LeftAuthority200324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound200367.actual selector witness) * (LeftAuthority200324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200371

namespace LeftBound200382
def owner : Owner := ⟨.program ⟨257⟩, ⟨21826⟩⟩
def transferEvent : Nat := 200382
def frameStart : Nat := 200280
def rule : BoundRule := .product (.predecessor 0 200380 .coefficient) (.predecessor 1 200381 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200380 .coefficient)
      LeftAuthority200335.bound (LeftAuthority200335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200381 .coefficient)
      LeftAuthority200378.bound (LeftAuthority200378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200378.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority200335.bound LeftAuthority200378.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200335.bound, LeftAuthority200378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority200335.actual selector witness) * (LeftAuthority200378.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200382

namespace LeftBound200390
def owner : Owner := ⟨.program ⟨257⟩, ⟨21827⟩⟩
def transferEvent : Nat := 200390
def frameStart : Nat := 200280
def rule : BoundRule := .sum [.predecessor 0 200388 .coefficient, .predecessor 1 200389 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200388 .coefficient)
      LeftAuthority200386.bound (LeftAuthority200386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200389 .coefficient)
      LeftBound200382.bound (LeftBound200382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200384RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200382.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200382.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority200386.bound, LeftBound200382.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200386.bound, LeftBound200382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority200386.actual selector witness, LeftBound200382.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200390

namespace LeftBound200394
def owner : Owner := ⟨.program ⟨257⟩, ⟨23465⟩⟩
def transferEvent : Nat := 200394
def frameStart : Nat := 200280
def rule : BoundRule := .sum [.predecessor 0 200392 .coefficient, .predecessor 1 200393 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200392 .coefficient)
      LeftBound200390.bound (LeftBound200390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200393 .coefficient)
      LeftBound200371.bound (LeftBound200371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200371.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound200390.bound, LeftBound200371.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200390.bound, LeftBound200371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound200390.actual selector witness, LeftBound200371.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200394

namespace LeftBound200407
def owner : Owner := ⟨.program ⟨257⟩, ⟨23463⟩⟩
def transferEvent : Nat := 200407
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 200405 .coefficient, .predecessor 1 200406 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200405 .coefficient)
      LeftBound200228.bound (LeftBound200228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200406 .coefficient)
      LeftBound200211.bound (LeftBound200211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200211.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound200228.bound, LeftBound200211.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200228.bound, LeftBound200211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound200228.actual selector witness, LeftBound200211.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200407

namespace LeftBound200410
def owner : Owner := ⟨.program ⟨257⟩, ⟨23463⟩⟩
def transferEvent : Nat := 200410
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 200404 .summary, .result 200218 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200404 .summary)
      LeftBound200230.bound (LeftBound200230.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22392⟩⟩) (rawTerms := some (Proof.Events782.exact200404RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound200230.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200218 .summary)
      LeftBound200213.bound (LeftBound200213.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23462⟩⟩) (rawTerms := some (Proof.Events782.exact200218RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound200213.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound200230.bound, LeftBound200213.bound]
def bound : CoeffClass := .finite ⟨2997834576566628384768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200230.bound, LeftBound200213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound200230.actual selector witness, LeftBound200213.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200410

namespace LeftBound200414
def owner : Owner := ⟨.program ⟨257⟩, ⟨23936⟩⟩
def transferEvent : Nat := 200414
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 200412 .coefficient) (.predecessor 1 200413 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200412 .coefficient)
      LeftBound200407.bound (LeftBound200407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200413 .coefficient)
      LeftAuthority200133.bound (LeftAuthority200133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events781.exact200134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200133.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200133.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound200407.bound LeftAuthority200133.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200407.bound, LeftAuthority200133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound200407.actual selector witness) * (LeftAuthority200133.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200414

namespace LeftBound200415
def owner : Owner := ⟨.program ⟨257⟩, ⟨23936⟩⟩
def transferEvent : Nat := 200415
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨23934⟩⟩]⟩ [⟨.result 200134 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200134 .coefficient)
      LeftAuthority200133.bound (LeftAuthority200133.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨23934⟩⟩) (rawTerms := some (Proof.Events781.exact200134RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200133.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200133.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority200133.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority200133.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound200415

namespace LeftBound200416
def owner : Owner := ⟨.program ⟨257⟩, ⟨23936⟩⟩
def transferEvent : Nat := 200416
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 200411 .summary) (.transfer 200415) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200411 .summary)
      LeftBound200410.bound (LeftBound200410.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23463⟩⟩) (rawTerms := some (Proof.Events782.exact200411RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound200410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 200415)
      LeftBound200415.bound (LeftBound200415.actual selector witness) := by
  exact .transfer (LeftBound200415.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound200410.bound LeftBound200415.bound
def bound : CoeffClass := .finite ⟨32189003662929192193909661368320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200410.bound, LeftBound200415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound200410.actual selector witness) * (LeftBound200415.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200416

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
