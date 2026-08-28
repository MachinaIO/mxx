import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1644
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1648
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1652
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1655
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1658

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound245393
def owner : Owner := ⟨.program ⟨257⟩, ⟨17200⟩⟩
def transferEvent : Nat := 245393
def frameStart : Nat := 245328
def rule : BoundRule := .product (.predecessor 0 245391 .coefficient) (.predecessor 1 245392 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245391 .coefficient)
      LeftAuthority245389.bound (LeftAuthority245389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority245389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority245389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245392 .coefficient)
      LeftBound245387.bound (LeftBound245387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245387.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245387.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority245389.bound LeftBound245387.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority245389.bound, LeftBound245387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority245389.actual selector witness) * (LeftBound245387.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound245393

namespace LeftBound245401
def owner : Owner := ⟨.program ⟨257⟩, ⟨17201⟩⟩
def transferEvent : Nat := 245401
def frameStart : Nat := 245328
def rule : BoundRule := .sum [.predecessor 0 245399 .coefficient, .predecessor 1 245400 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245399 .coefficient)
      LeftAuthority245397.bound (LeftAuthority245397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority245397.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority245397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245400 .coefficient)
      LeftBound245393.bound (LeftBound245393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245393.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority245397.bound, LeftBound245393.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority245397.bound, LeftBound245393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority245397.actual selector witness, LeftBound245393.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245401

namespace LeftBound245405
def owner : Owner := ⟨.program ⟨257⟩, ⟨17706⟩⟩
def transferEvent : Nat := 245405
def frameStart : Nat := 245328
def rule : BoundRule := .product (.predecessor 0 245403 .coefficient) (.predecessor 1 245404 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245403 .coefficient)
      LeftBound245401.bound (LeftBound245401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245404 .coefficient)
      LeftAuthority245378.bound (LeftAuthority245378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority245378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority245378.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound245401.bound LeftAuthority245378.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245401.bound, LeftAuthority245378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound245401.actual selector witness) * (LeftAuthority245378.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound245405

namespace LeftBound245416
def owner : Owner := ⟨.program ⟨257⟩, ⟨16004⟩⟩
def transferEvent : Nat := 245416
def frameStart : Nat := 245328
def rule : BoundRule := .product (.predecessor 0 245414 .coefficient) (.predecessor 1 245415 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245414 .coefficient)
      LeftAuthority245389.bound (LeftAuthority245389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority245389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority245389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245415 .coefficient)
      LeftAuthority245412.bound (LeftAuthority245412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority245412.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority245412.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority245389.bound LeftAuthority245412.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority245389.bound, LeftAuthority245412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority245389.actual selector witness) * (LeftAuthority245412.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound245416

namespace LeftBound245424
def owner : Owner := ⟨.program ⟨257⟩, ⟨16005⟩⟩
def transferEvent : Nat := 245424
def frameStart : Nat := 245328
def rule : BoundRule := .sum [.predecessor 0 245422 .coefficient, .predecessor 1 245423 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245422 .coefficient)
      LeftAuthority245420.bound (LeftAuthority245420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority245420.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority245420.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245423 .coefficient)
      LeftBound245416.bound (LeftBound245416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority245420.bound, LeftBound245416.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority245420.bound, LeftBound245416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority245420.actual selector witness, LeftBound245416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245424

namespace LeftBound245428
def owner : Owner := ⟨.program ⟨257⟩, ⟨17709⟩⟩
def transferEvent : Nat := 245428
def frameStart : Nat := 245328
def rule : BoundRule := .sum [.predecessor 0 245426 .coefficient, .predecessor 1 245427 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245426 .coefficient)
      LeftBound245424.bound (LeftBound245424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245424.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245424.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245427 .coefficient)
      LeftBound245405.bound (LeftBound245405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245405.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245405.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245424.bound, LeftBound245405.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245424.bound, LeftBound245405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245424.actual selector witness, LeftBound245405.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245428

namespace LeftBound245441
def owner : Owner := ⟨.program ⟨257⟩, ⟨17708⟩⟩
def transferEvent : Nat := 245441
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245439 .coefficient, .predecessor 1 245440 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245439 .coefficient)
      LeftBound245270.bound (LeftBound245270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245440 .coefficient)
      LeftBound245253.bound (LeftBound245253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245253.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245270.bound, LeftBound245253.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245270.bound, LeftBound245253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245270.actual selector witness, LeftBound245253.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245441

namespace LeftBound245444
def owner : Owner := ⟨.program ⟨257⟩, ⟨17708⟩⟩
def transferEvent : Nat := 245444
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245438 .summary, .result 245260 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245438 .summary)
      LeftBound245272.bound (LeftBound245272.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16559⟩⟩) (rawTerms := some (Proof.Events958.exact245438RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245260 .summary)
      LeftBound245255.bound (LeftBound245255.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17707⟩⟩) (rawTerms := some (Proof.Events958.exact245260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245255.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245272.bound, LeftBound245255.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245272.bound, LeftBound245255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245272.actual selector witness, LeftBound245255.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245444

namespace LeftBound245448
def owner : Owner := ⟨.program ⟨257⟩, ⟨20594⟩⟩
def transferEvent : Nat := 245448
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245446 .coefficient, .predecessor 1 245447 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245446 .coefficient)
      LeftBound245441.bound (LeftBound245441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245441.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245447 .coefficient)
      LeftBound244959.bound (LeftBound244959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events956.exact244963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244959.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245441.bound, LeftBound244959.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245441.bound, LeftBound244959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245441.actual selector witness, LeftBound244959.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245448

namespace LeftBound245449
def owner : Owner := ⟨.program ⟨257⟩, ⟨20594⟩⟩
def transferEvent : Nat := 245449
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245445 .summary, .result 244963 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245445 .summary)
      LeftBound245444.bound (LeftBound245444.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17708⟩⟩) (rawTerms := some (Proof.Events958.exact245445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 244963 .summary)
      LeftBound244962.bound (LeftBound244962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20593⟩⟩) (rawTerms := some (Proof.Events956.exact244963RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound244962.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245444.bound, LeftBound244962.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245444.bound, LeftBound244962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245444.actual selector witness, LeftBound244962.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245449

namespace LeftBound245453
def owner : Owner := ⟨.program ⟨257⟩, ⟨23814⟩⟩
def transferEvent : Nat := 245453
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245451 .coefficient, .predecessor 1 245452 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245451 .coefficient)
      LeftBound245448.bound (LeftBound245448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245448.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245452 .coefficient)
      LeftBound244477.bound (LeftBound244477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events955.exact244481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245448.bound, LeftBound244477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245448.bound, LeftBound244477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245448.actual selector witness, LeftBound244477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245453

namespace LeftBound245454
def owner : Owner := ⟨.program ⟨257⟩, ⟨23814⟩⟩
def transferEvent : Nat := 245454
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245450 .summary, .result 244481 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245450 .summary)
      LeftBound245449.bound (LeftBound245449.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20594⟩⟩) (rawTerms := some (Proof.Events958.exact245450RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245449.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 244481 .summary)
      LeftBound244480.bound (LeftBound244480.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23813⟩⟩) (rawTerms := some (Proof.Events955.exact244481RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound244480.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245449.bound, LeftBound244480.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245449.bound, LeftBound244480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245449.actual selector witness, LeftBound244480.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245454

namespace LeftBound245458
def owner : Owner := ⟨.program ⟨257⟩, ⟨33834⟩⟩
def transferEvent : Nat := 245458
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245456 .coefficient, .predecessor 1 245457 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245456 .coefficient)
      LeftBound245453.bound (LeftBound245453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245453.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245453.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245457 .coefficient)
      LeftBound243995.bound (LeftBound243995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact243999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245453.bound, LeftBound243995.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245453.bound, LeftBound243995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245453.actual selector witness, LeftBound243995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245458

namespace LeftBound245459
def owner : Owner := ⟨.program ⟨257⟩, ⟨33834⟩⟩
def transferEvent : Nat := 245459
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245455 .summary, .result 243999 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245455 .summary)
      LeftBound245454.bound (LeftBound245454.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23814⟩⟩) (rawTerms := some (Proof.Events958.exact245455RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243999 .summary)
      LeftBound243998.bound (LeftBound243998.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33833⟩⟩) (rawTerms := some (Proof.Events953.exact243999RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound243998.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245454.bound, LeftBound243998.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245454.bound, LeftBound243998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245454.actual selector witness, LeftBound243998.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245459

namespace LeftBound245463
def owner : Owner := ⟨.program ⟨257⟩, ⟨52894⟩⟩
def transferEvent : Nat := 245463
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245461 .coefficient, .predecessor 1 245462 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245461 .coefficient)
      LeftBound245458.bound (LeftBound245458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events958.exact245460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245462 .coefficient)
      LeftBound243513.bound (LeftBound243513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243513.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245458.bound, LeftBound243513.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245458.bound, LeftBound243513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245458.actual selector witness, LeftBound243513.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245463

namespace LeftBound245464
def owner : Owner := ⟨.program ⟨257⟩, ⟨52894⟩⟩
def transferEvent : Nat := 245464
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245460 .summary, .result 243517 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245460 .summary)
      LeftBound245459.bound (LeftBound245459.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33834⟩⟩) (rawTerms := some (Proof.Events958.exact245460RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243517 .summary)
      LeftBound243516.bound (LeftBound243516.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52893⟩⟩) (rawTerms := some (Proof.Events951.exact243517RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound243516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245459.bound, LeftBound243516.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245459.bound, LeftBound243516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245459.actual selector witness, LeftBound243516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245464

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
