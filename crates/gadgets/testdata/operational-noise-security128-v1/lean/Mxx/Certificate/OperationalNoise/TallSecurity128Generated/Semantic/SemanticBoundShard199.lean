import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard094
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard171
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard173
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard198

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35341
def owner : Owner := ⟨.program ⟨257⟩, ⟨30484⟩⟩
def transferEvent : Nat := 35341
def frameStart : Nat := 35276
def rule : BoundRule := .product (.predecessor 0 35339 .coefficient) (.predecessor 1 35340 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35339 .coefficient)
      LeftAuthority35337.bound (LeftAuthority35337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35337.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35340 .coefficient)
      LeftBound35335.bound (LeftBound35335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35335.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority35337.bound LeftBound35335.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35337.bound, LeftBound35335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority35337.actual selector witness) * (LeftBound35335.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35341

namespace LeftBound35349
def owner : Owner := ⟨.program ⟨257⟩, ⟨30485⟩⟩
def transferEvent : Nat := 35349
def frameStart : Nat := 35276
def rule : BoundRule := .sum [.predecessor 0 35347 .coefficient, .predecessor 1 35348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35347 .coefficient)
      LeftAuthority35345.bound (LeftAuthority35345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35345.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35348 .coefficient)
      LeftBound35341.bound (LeftBound35341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35341.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority35345.bound, LeftBound35341.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35345.bound, LeftBound35341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority35345.actual selector witness, LeftBound35341.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35349

namespace LeftBound35353
def owner : Owner := ⟨.program ⟨257⟩, ⟨31195⟩⟩
def transferEvent : Nat := 35353
def frameStart : Nat := 35276
def rule : BoundRule := .product (.predecessor 0 35351 .coefficient) (.predecessor 1 35352 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35351 .coefficient)
      LeftBound35349.bound (LeftBound35349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35349.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35352 .coefficient)
      LeftAuthority35326.bound (LeftAuthority35326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35326.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound35349.bound LeftAuthority35326.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35349.bound, LeftAuthority35326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound35349.actual selector witness) * (LeftAuthority35326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35353

namespace LeftBound35364
def owner : Owner := ⟨.program ⟨257⟩, ⟨29417⟩⟩
def transferEvent : Nat := 35364
def frameStart : Nat := 35276
def rule : BoundRule := .product (.predecessor 0 35362 .coefficient) (.predecessor 1 35363 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35362 .coefficient)
      LeftAuthority35337.bound (LeftAuthority35337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35337.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35363 .coefficient)
      LeftAuthority35360.bound (LeftAuthority35360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority35337.bound LeftAuthority35360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35337.bound, LeftAuthority35360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority35337.actual selector witness) * (LeftAuthority35360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35364

namespace LeftBound35372
def owner : Owner := ⟨.program ⟨257⟩, ⟨29418⟩⟩
def transferEvent : Nat := 35372
def frameStart : Nat := 35276
def rule : BoundRule := .sum [.predecessor 0 35370 .coefficient, .predecessor 1 35371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35370 .coefficient)
      LeftAuthority35368.bound (LeftAuthority35368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35371 .coefficient)
      LeftBound35364.bound (LeftBound35364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority35368.bound, LeftBound35364.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35368.bound, LeftBound35364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority35368.actual selector witness, LeftBound35364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35372

namespace LeftBound35376
def owner : Owner := ⟨.program ⟨257⟩, ⟨31198⟩⟩
def transferEvent : Nat := 35376
def frameStart : Nat := 35276
def rule : BoundRule := .sum [.predecessor 0 35374 .coefficient, .predecessor 1 35375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35374 .coefficient)
      LeftBound35372.bound (LeftBound35372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35375 .coefficient)
      LeftBound35353.bound (LeftBound35353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35372.bound, LeftBound35353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35372.bound, LeftBound35353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound35372.actual selector witness, LeftBound35353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35376

namespace LeftBound35389
def owner : Owner := ⟨.program ⟨257⟩, ⟨31197⟩⟩
def transferEvent : Nat := 35389
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35387 .coefficient, .predecessor 1 35388 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35387 .coefficient)
      LeftBound35218.bound (LeftBound35218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35388 .coefficient)
      LeftBound35201.bound (LeftBound35201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35218.bound, LeftBound35201.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35218.bound, LeftBound35201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound35218.actual selector witness, LeftBound35201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35389

namespace LeftBound35392
def owner : Owner := ⟨.program ⟨257⟩, ⟨31197⟩⟩
def transferEvent : Nat := 35392
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35386 .summary, .result 35208 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 35386 .summary)
      LeftBound35220.bound (LeftBound35220.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30019⟩⟩) (rawTerms := some (Proof.Events138.exact35386RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35220.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 35208 .summary)
      LeftBound35203.bound (LeftBound35203.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31196⟩⟩) (rawTerms := some (Proof.Events137.exact35208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35203.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35220.bound, LeftBound35203.bound]
def bound : CoeffClass := .finite ⟨32192146870060392302605751287808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35220.bound, LeftBound35203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound35220.actual selector witness, LeftBound35203.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35392

namespace LeftBound35416
def owner : Owner := ⟨.program ⟨257⟩, ⟨26313⟩⟩
def transferEvent : Nat := 35416
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 35414 .coefficient) (.predecessor 1 35415 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35414 .coefficient)
      LeftAuthority1002.bound (LeftAuthority1002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact1003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35415 .coefficient)
      LeftBound32026.bound (LeftBound32026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority1002.bound LeftBound32026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1002.bound, LeftBound32026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority1002.actual selector witness) * (LeftBound32026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound35416

namespace LeftBound35421
def owner : Owner := ⟨.program ⟨257⟩, ⟨11611⟩⟩
def transferEvent : Nat := 35421
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35419 .coefficient) (.predecessor 1 35420 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35419 .coefficient)
      LeftBound31897.bound (LeftBound31897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35420 .coefficient)
      LeftBound20586.bound (LeftBound20586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20586.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound31897.bound LeftBound20586.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31897.bound, LeftBound20586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound31897.actual selector witness) * (LeftBound20586.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35421

namespace LeftBound35426
def owner : Owner := ⟨.program ⟨257⟩, ⟨26314⟩⟩
def transferEvent : Nat := 35426
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35424 .coefficient, .predecessor 1 35425 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35424 .coefficient)
      LeftBound35421.bound (LeftBound35421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35425 .coefficient)
      LeftBound35416.bound (LeftBound35416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35421.bound, LeftBound35416.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35421.bound, LeftBound35416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound35421.actual selector witness, LeftBound35416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35426

namespace LeftBound35430
def owner : Owner := ⟨.program ⟨257⟩, ⟨26315⟩⟩
def transferEvent : Nat := 35430
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35428 .coefficient, .predecessor 1 35429 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35428 .coefficient)
      LeftBound35426.bound (LeftBound35426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35429 .coefficient)
      LeftBound20578.bound (LeftBound20578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20578.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35426.bound, LeftBound20578.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35426.bound, LeftBound20578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound35426.actual selector witness, LeftBound20578.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35430

namespace LeftBound35431
def owner : Owner := ⟨.program ⟨257⟩, ⟨26315⟩⟩
def transferEvent : Nat := 35431
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨104⟩⟩]⟩ [⟨.result 20579 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20579 .coefficient)
      LeftBound20578.bound (LeftBound20578.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨104⟩⟩) (rawTerms := some (Proof.Events080.exact20579RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20578.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20578.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound20578.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35431

namespace LeftBound35436
def owner : Owner := ⟨.program ⟨257⟩, ⟨26316⟩⟩
def transferEvent : Nat := 35436
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35434 .coefficient) (.predecessor 1 35435 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 35434 .coefficient)
      LeftBound35430.bound (LeftBound35430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 35435 .coefficient)
      LeftAuthority1005.bound (LeftAuthority1005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact1006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1005.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1005.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound35430.bound LeftAuthority1005.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35430.bound, LeftAuthority1005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound35430.actual selector witness) * (LeftAuthority1005.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35436

namespace LeftBound35437
def owner : Owner := ⟨.program ⟨257⟩, ⟨26316⟩⟩
def transferEvent : Nat := 35437
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13116⟩⟩], []⟩ [⟨.result 1006 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 1006 .coefficient)
      LeftAuthority1005.bound (LeftAuthority1005.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13116⟩⟩) (rawTerms := some (Proof.Events003.exact1006RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1005.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1005.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1005.bound []
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority1005.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35437

namespace LeftBound35438
def owner : Owner := ⟨.program ⟨257⟩, ⟨26316⟩⟩
def transferEvent : Nat := 35438
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 35433 .summary) (.transfer 35437) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 35433 .summary)
      LeftBound35431.bound (LeftBound35431.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26315⟩⟩) (rawTerms := some (Proof.Events138.exact35433RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 35437)
      LeftBound35437.bound (LeftBound35437.actual selector witness) := by
  exact .transfer (LeftBound35437.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound35431.bound LeftBound35437.bound
def bound : CoeffClass := .finite ⟨25559040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35431.bound, LeftBound35437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound35431.actual selector witness) * (LeftBound35437.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35438

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
