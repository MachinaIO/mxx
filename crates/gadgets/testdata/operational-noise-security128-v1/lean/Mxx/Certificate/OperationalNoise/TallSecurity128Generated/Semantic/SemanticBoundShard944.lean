import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard885
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard943

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound142321
def owner : Owner := ⟨.program ⟨257⟩, ⟨19960⟩⟩
def transferEvent : Nat := 142321
def frameStart : Nat := 142262
def rule : BoundRule := .product (.predecessor 0 142319 .coefficient) (.predecessor 1 142320 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142319 .coefficient)
      LeftAuthority142317.bound (LeftAuthority142317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142317.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142320 .coefficient)
      LeftBound142315.bound (LeftBound142315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142315.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority142317.bound LeftBound142315.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142317.bound, LeftBound142315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority142317.actual selector witness) * (LeftBound142315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142321

namespace LeftBound142337
def owner : Owner := ⟨.program ⟨257⟩, ⟨9572⟩⟩
def transferEvent : Nat := 142337
def frameStart : Nat := 142262
def rule : BoundRule := .scale (.predecessor 0 142335 .coefficient) (.value (.predecessor 1 142336 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142335 .coefficient)
      LeftAuthority142333.bound (LeftAuthority142333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142333.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142336 .coefficient)
      LeftAuthority142324.bound (LeftAuthority142324.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority142324.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority142333.bound LeftAuthority142324.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142333.bound, LeftAuthority142324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority142333.actual selector witness) * (LeftAuthority142324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound142337

namespace LeftBound142340
def owner : Owner := ⟨.program ⟨257⟩, ⟨7277⟩⟩
def transferEvent : Nat := 142340
def frameStart : Nat := 142262
def rule : BoundRule := .identity (.predecessor 0 142339 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142339 .coefficient)
      LeftAuthority142327.bound (LeftAuthority142327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142327.derived selector witness)

def rawBound : CoeffClass := LeftAuthority142327.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority142327.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound142340

namespace LeftBound142344
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def transferEvent : Nat := 142344
def frameStart : Nat := 142262
def rule : BoundRule := .product (.predecessor 0 142342 .coefficient) (.predecessor 1 142343 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142342 .coefficient)
      LeftBound142340.bound (LeftBound142340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142343 .coefficient)
      LeftBound142337.bound (LeftBound142337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142337.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound142340.bound LeftBound142337.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142340.bound, LeftBound142337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound142340.actual selector witness) * (LeftBound142337.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142344

namespace LeftBound142349
def owner : Owner := ⟨.program ⟨257⟩, ⟨19961⟩⟩
def transferEvent : Nat := 142349
def frameStart : Nat := 142262
def rule : BoundRule := .sum [.predecessor 0 142347 .coefficient, .predecessor 1 142348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142347 .coefficient)
      LeftBound142344.bound (LeftBound142344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142344.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142348 .coefficient)
      LeftBound142321.bound (LeftBound142321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142321.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142344.bound, LeftBound142321.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142344.bound, LeftBound142321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142344.actual selector witness, LeftBound142321.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142349

namespace LeftBound142353
def owner : Owner := ⟨.program ⟨257⟩, ⟨20145⟩⟩
def transferEvent : Nat := 142353
def frameStart : Nat := 142262
def rule : BoundRule := .product (.predecessor 0 142351 .coefficient) (.predecessor 1 142352 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142351 .coefficient)
      LeftBound142349.bound (LeftBound142349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142349.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142352 .coefficient)
      LeftAuthority142306.bound (LeftAuthority142306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound142349.bound LeftAuthority142306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142349.bound, LeftAuthority142306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound142349.actual selector witness) * (LeftAuthority142306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142353

namespace LeftBound142364
def owner : Owner := ⟨.program ⟨257⟩, ⟨18534⟩⟩
def transferEvent : Nat := 142364
def frameStart : Nat := 142262
def rule : BoundRule := .product (.predecessor 0 142362 .coefficient) (.predecessor 1 142363 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142362 .coefficient)
      LeftAuthority142317.bound (LeftAuthority142317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142317.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142363 .coefficient)
      LeftAuthority142360.bound (LeftAuthority142360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority142317.bound LeftAuthority142360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142317.bound, LeftAuthority142360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority142317.actual selector witness) * (LeftAuthority142360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142364

namespace LeftBound142372
def owner : Owner := ⟨.program ⟨257⟩, ⟨18535⟩⟩
def transferEvent : Nat := 142372
def frameStart : Nat := 142262
def rule : BoundRule := .sum [.predecessor 0 142370 .coefficient, .predecessor 1 142371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142370 .coefficient)
      LeftAuthority142368.bound (LeftAuthority142368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142371 .coefficient)
      LeftBound142364.bound (LeftBound142364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority142368.bound, LeftBound142364.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142368.bound, LeftBound142364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority142368.actual selector witness, LeftBound142364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142372

namespace LeftBound142376
def owner : Owner := ⟨.program ⟨257⟩, ⟨20146⟩⟩
def transferEvent : Nat := 142376
def frameStart : Nat := 142262
def rule : BoundRule := .sum [.predecessor 0 142374 .coefficient, .predecessor 1 142375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142374 .coefficient)
      LeftBound142372.bound (LeftBound142372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142375 .coefficient)
      LeftBound142353.bound (LeftBound142353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142372.bound, LeftBound142353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142372.bound, LeftBound142353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142372.actual selector witness, LeftBound142353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142376

namespace LeftBound142389
def owner : Owner := ⟨.program ⟨257⟩, ⟨20144⟩⟩
def transferEvent : Nat := 142389
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 142387 .coefficient, .predecessor 1 142388 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142387 .coefficient)
      LeftBound142210.bound (LeftBound142210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142388 .coefficient)
      LeftBound142193.bound (LeftBound142193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142193.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142210.bound, LeftBound142193.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142210.bound, LeftBound142193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142210.actual selector witness, LeftBound142193.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142389

namespace LeftBound142392
def owner : Owner := ⟨.program ⟨257⟩, ⟨20144⟩⟩
def transferEvent : Nat := 142392
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 142386 .summary, .result 142200 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142386 .summary)
      LeftBound142212.bound (LeftBound142212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19082⟩⟩) (rawTerms := some (Proof.Events556.exact142386RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound142212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142200 .summary)
      LeftBound142195.bound (LeftBound142195.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20143⟩⟩) (rawTerms := some (Proof.Events555.exact142200RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound142195.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142212.bound, LeftBound142195.bound]
def bound : CoeffClass := .finite ⟨2997825428629885288448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142212.bound, LeftBound142195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142212.actual selector witness, LeftBound142195.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142392

namespace LeftBound142396
def owner : Owner := ⟨.program ⟨257⟩, ⟨20437⟩⟩
def transferEvent : Nat := 142396
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 142394 .coefficient) (.predecessor 1 142395 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142394 .coefficient)
      LeftBound142389.bound (LeftBound142389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142395 .coefficient)
      LeftAuthority142115.bound (LeftAuthority142115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142115.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound142389.bound LeftAuthority142115.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142389.bound, LeftAuthority142115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound142389.actual selector witness) * (LeftAuthority142115.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142396

namespace LeftBound142397
def owner : Owner := ⟨.program ⟨257⟩, ⟨20437⟩⟩
def transferEvent : Nat := 142397
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨20435⟩⟩]⟩ [⟨.result 142116 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142116 .coefficient)
      LeftAuthority142115.bound (LeftAuthority142115.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨20435⟩⟩) (rawTerms := some (Proof.Events555.exact142116RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142115.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority142115.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority142115.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound142397

namespace LeftBound142398
def owner : Owner := ⟨.program ⟨257⟩, ⟨20437⟩⟩
def transferEvent : Nat := 142398
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 142393 .summary) (.transfer 142397) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142393 .summary)
      LeftBound142392.bound (LeftBound142392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20144⟩⟩) (rawTerms := some (Proof.Events556.exact142393RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound142392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 142397)
      LeftBound142397.bound (LeftBound142397.actual selector witness) := by
  exact .transfer (LeftBound142397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound142392.bound LeftBound142397.bound
def bound : CoeffClass := .finite ⟨32188905437706348505289216491520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142392.bound, LeftBound142397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound142392.actual selector witness) * (LeftBound142397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142398

namespace LeftBound142409
def owner : Owner := ⟨.program ⟨257⟩, ⟨19318⟩⟩
def transferEvent : Nat := 142409
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 142407 .coefficient) (.value (.predecessor 1 142408 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142407 .coefficient)
      LeftAuthority142405.bound (LeftAuthority142405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142408 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority142405.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142405.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority142405.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound142409

namespace LeftBound142413
def owner : Owner := ⟨.program ⟨257⟩, ⟨19319⟩⟩
def transferEvent : Nat := 142413
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 142411 .coefficient) (.predecessor 1 142412 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142411 .coefficient)
      LeftBound134492.bound (LeftBound134492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events525.exact134495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142412 .coefficient)
      LeftBound142409.bound (LeftBound142409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142409.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound134492.bound LeftBound142409.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134492.bound, LeftBound142409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound134492.actual selector witness) * (LeftBound142409.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142413

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
