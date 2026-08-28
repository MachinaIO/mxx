import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1642

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound243224
def owner : Owner := ⟨.program ⟨257⟩, ⟨50493⟩⟩
def transferEvent : Nat := 243224
def frameStart : Nat := 243191
def rule : BoundRule := .identity (.predecessor 0 243223 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243223 .coefficient)
      LeftBound243220.bound (LeftBound243220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243220.derived selector witness)

def rawBound : CoeffClass := LeftBound243220.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound243220.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound243224

namespace LeftBound243241
def owner : Owner := ⟨.program ⟨257⟩, ⟨52278⟩⟩
def transferEvent : Nat := 243241
def frameStart : Nat := 243191
def rule : BoundRule := .sum [.predecessor 0 243239 .coefficient, .predecessor 1 243240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243239 .coefficient)
      LeftBound243224.bound (LeftBound243224.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound243224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243240 .coefficient)
      LeftAuthority243237.bound (LeftAuthority243237.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority243237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243224.bound, LeftAuthority243237.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243224.bound, LeftAuthority243237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243224.actual selector witness, LeftAuthority243237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243241

namespace LeftBound243244
def owner : Owner := ⟨.program ⟨257⟩, ⟨52279⟩⟩
def transferEvent : Nat := 243244
def frameStart : Nat := 243191
def rule : BoundRule := .identity (.predecessor 0 243243 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243243 .coefficient)
      LeftBound243241.bound (LeftBound243241.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound243241.derived selector witness)

def rawBound : CoeffClass := LeftBound243241.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound243241.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound243244

namespace LeftBound243250
def owner : Owner := ⟨.program ⟨257⟩, ⟨52280⟩⟩
def transferEvent : Nat := 243250
def frameStart : Nat := 243191
def rule : BoundRule := .product (.predecessor 0 243248 .coefficient) (.predecessor 1 243249 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243248 .coefficient)
      LeftAuthority243246.bound (LeftAuthority243246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243249 .coefficient)
      LeftBound243244.bound (LeftBound243244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority243246.bound LeftBound243244.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243246.bound, LeftBound243244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority243246.actual selector witness) * (LeftBound243244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243250

namespace LeftBound243266
def owner : Owner := ⟨.program ⟨257⟩, ⟨9581⟩⟩
def transferEvent : Nat := 243266
def frameStart : Nat := 243191
def rule : BoundRule := .scale (.predecessor 0 243264 .coefficient) (.value (.predecessor 1 243265 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243264 .coefficient)
      LeftAuthority243262.bound (LeftAuthority243262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243265 .coefficient)
      LeftAuthority243253.bound (LeftAuthority243253.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority243253.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority243262.bound LeftAuthority243253.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243262.bound, LeftAuthority243253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority243262.actual selector witness) * (LeftAuthority243253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound243266

namespace LeftBound243269
def owner : Owner := ⟨.program ⟨257⟩, ⟨7288⟩⟩
def transferEvent : Nat := 243269
def frameStart : Nat := 243191
def rule : BoundRule := .identity (.predecessor 0 243268 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243268 .coefficient)
      LeftAuthority243256.bound (LeftAuthority243256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243256.derived selector witness)

def rawBound : CoeffClass := LeftAuthority243256.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority243256.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound243269

namespace LeftBound243273
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def transferEvent : Nat := 243273
def frameStart : Nat := 243191
def rule : BoundRule := .product (.predecessor 0 243271 .coefficient) (.predecessor 1 243272 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243271 .coefficient)
      LeftBound243269.bound (LeftBound243269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243269.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243272 .coefficient)
      LeftBound243266.bound (LeftBound243266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound243269.bound LeftBound243266.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243269.bound, LeftBound243266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound243269.actual selector witness) * (LeftBound243266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243273

namespace LeftBound243278
def owner : Owner := ⟨.program ⟨257⟩, ⟨52281⟩⟩
def transferEvent : Nat := 243278
def frameStart : Nat := 243191
def rule : BoundRule := .sum [.predecessor 0 243276 .coefficient, .predecessor 1 243277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243276 .coefficient)
      LeftBound243273.bound (LeftBound243273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243277 .coefficient)
      LeftBound243250.bound (LeftBound243250.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243250.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243250.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243273.bound, LeftBound243250.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243273.bound, LeftBound243250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243273.actual selector witness, LeftBound243250.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243278

namespace LeftBound243282
def owner : Owner := ⟨.program ⟨257⟩, ⟨52500⟩⟩
def transferEvent : Nat := 243282
def frameStart : Nat := 243191
def rule : BoundRule := .product (.predecessor 0 243280 .coefficient) (.predecessor 1 243281 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243280 .coefficient)
      LeftBound243278.bound (LeftBound243278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243281 .coefficient)
      LeftAuthority243235.bound (LeftAuthority243235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound243278.bound LeftAuthority243235.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243278.bound, LeftAuthority243235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound243278.actual selector witness) * (LeftAuthority243235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243282

namespace LeftBound243293
def owner : Owner := ⟨.program ⟨257⟩, ⟨50874⟩⟩
def transferEvent : Nat := 243293
def frameStart : Nat := 243191
def rule : BoundRule := .product (.predecessor 0 243291 .coefficient) (.predecessor 1 243292 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243291 .coefficient)
      LeftAuthority243246.bound (LeftAuthority243246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243292 .coefficient)
      LeftAuthority243289.bound (LeftAuthority243289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority243246.bound LeftAuthority243289.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243246.bound, LeftAuthority243289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority243246.actual selector witness) * (LeftAuthority243289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243293

namespace LeftBound243301
def owner : Owner := ⟨.program ⟨257⟩, ⟨50875⟩⟩
def transferEvent : Nat := 243301
def frameStart : Nat := 243191
def rule : BoundRule := .sum [.predecessor 0 243299 .coefficient, .predecessor 1 243300 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243299 .coefficient)
      LeftAuthority243297.bound (LeftAuthority243297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243297.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243300 .coefficient)
      LeftBound243293.bound (LeftBound243293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243293.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority243297.bound, LeftBound243293.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243297.bound, LeftBound243293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority243297.actual selector witness, LeftBound243293.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243301

namespace LeftBound243305
def owner : Owner := ⟨.program ⟨257⟩, ⟨52501⟩⟩
def transferEvent : Nat := 243305
def frameStart : Nat := 243191
def rule : BoundRule := .sum [.predecessor 0 243303 .coefficient, .predecessor 1 243304 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243303 .coefficient)
      LeftBound243301.bound (LeftBound243301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243304 .coefficient)
      LeftBound243282.bound (LeftBound243282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243301.bound, LeftBound243282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243301.bound, LeftBound243282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243301.actual selector witness, LeftBound243282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243305

namespace LeftBound243318
def owner : Owner := ⟨.program ⟨257⟩, ⟨52499⟩⟩
def transferEvent : Nat := 243318
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 243316 .coefficient, .predecessor 1 243317 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243316 .coefficient)
      LeftBound243139.bound (LeftBound243139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243317 .coefficient)
      LeftBound243122.bound (LeftBound243122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243122.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243139.bound, LeftBound243122.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243139.bound, LeftBound243122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243139.actual selector witness, LeftBound243122.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243318

namespace LeftBound243321
def owner : Owner := ⟨.program ⟨257⟩, ⟨52499⟩⟩
def transferEvent : Nat := 243321
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 243315 .summary, .result 243129 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243315 .summary)
      LeftBound243141.bound (LeftBound243141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51432⟩⟩) (rawTerms := some (Proof.Events950.exact243315RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound243141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243129 .summary)
      LeftBound243124.bound (LeftBound243124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52498⟩⟩) (rawTerms := some (Proof.Events949.exact243129RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound243124.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243141.bound, LeftBound243124.bound]
def bound : CoeffClass := .finite ⟨2997889464187086962688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243141.bound, LeftBound243124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243141.actual selector witness, LeftBound243124.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243321

namespace LeftBound243325
def owner : Owner := ⟨.program ⟨257⟩, ⟨52892⟩⟩
def transferEvent : Nat := 243325
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 243323 .coefficient) (.predecessor 1 243324 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243323 .coefficient)
      LeftBound243318.bound (LeftBound243318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243324 .coefficient)
      LeftAuthority243044.bound (LeftAuthority243044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243044.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243044.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound243318.bound LeftAuthority243044.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243318.bound, LeftAuthority243044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound243318.actual selector witness) * (LeftAuthority243044.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243325

namespace LeftBound243326
def owner : Owner := ⟨.program ⟨257⟩, ⟨52892⟩⟩
def transferEvent : Nat := 243326
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨52890⟩⟩]⟩ [⟨.result 243045 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243045 .coefficient)
      LeftAuthority243044.bound (LeftAuthority243044.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨52890⟩⟩) (rawTerms := some (Proof.Events949.exact243045RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243044.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243044.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority243044.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority243044.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound243326

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
