import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard070
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard071
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1286
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1289
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1292

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound193318
def owner : Owner := ⟨.program ⟨257⟩, ⟨49515⟩⟩
def transferEvent : Nat := 193318
def frameStart : Nat := 193259
def rule : BoundRule := .identity (.predecessor 0 193317 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193317 .coefficient)
      LeftBound193315.bound (LeftBound193315.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound193315.derived selector witness)

def rawBound : CoeffClass := LeftBound193315.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound193315.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound193318

namespace LeftBound193324
def owner : Owner := ⟨.program ⟨257⟩, ⟨49516⟩⟩
def transferEvent : Nat := 193324
def frameStart : Nat := 193259
def rule : BoundRule := .product (.predecessor 0 193322 .coefficient) (.predecessor 1 193323 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193322 .coefficient)
      LeftAuthority193320.bound (LeftAuthority193320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193323 .coefficient)
      LeftBound193318.bound (LeftBound193318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193318.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority193320.bound LeftBound193318.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193320.bound, LeftBound193318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority193320.actual selector witness) * (LeftBound193318.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193324

namespace LeftBound193332
def owner : Owner := ⟨.program ⟨257⟩, ⟨49517⟩⟩
def transferEvent : Nat := 193332
def frameStart : Nat := 193259
def rule : BoundRule := .sum [.predecessor 0 193330 .coefficient, .predecessor 1 193331 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193330 .coefficient)
      LeftAuthority193328.bound (LeftAuthority193328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193328.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193331 .coefficient)
      LeftBound193324.bound (LeftBound193324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193324.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority193328.bound, LeftBound193324.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193328.bound, LeftBound193324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority193328.actual selector witness, LeftBound193324.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193332

namespace LeftBound193336
def owner : Owner := ⟨.program ⟨257⟩, ⟨50080⟩⟩
def transferEvent : Nat := 193336
def frameStart : Nat := 193259
def rule : BoundRule := .product (.predecessor 0 193334 .coefficient) (.predecessor 1 193335 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193334 .coefficient)
      LeftBound193332.bound (LeftBound193332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193332.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193335 .coefficient)
      LeftAuthority193309.bound (LeftAuthority193309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193309.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193309.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound193332.bound LeftAuthority193309.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193332.bound, LeftAuthority193309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound193332.actual selector witness) * (LeftAuthority193309.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193336

namespace LeftBound193347
def owner : Owner := ⟨.program ⟨257⟩, ⟨48390⟩⟩
def transferEvent : Nat := 193347
def frameStart : Nat := 193259
def rule : BoundRule := .product (.predecessor 0 193345 .coefficient) (.predecessor 1 193346 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193345 .coefficient)
      LeftAuthority193320.bound (LeftAuthority193320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193346 .coefficient)
      LeftAuthority193343.bound (LeftAuthority193343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193343.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193343.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority193320.bound LeftAuthority193343.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193320.bound, LeftAuthority193343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority193320.actual selector witness) * (LeftAuthority193343.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193347

namespace LeftBound193355
def owner : Owner := ⟨.program ⟨257⟩, ⟨48391⟩⟩
def transferEvent : Nat := 193355
def frameStart : Nat := 193259
def rule : BoundRule := .sum [.predecessor 0 193353 .coefficient, .predecessor 1 193354 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193353 .coefficient)
      LeftAuthority193351.bound (LeftAuthority193351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193351.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193354 .coefficient)
      LeftBound193347.bound (LeftBound193347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193347.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority193351.bound, LeftBound193347.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193351.bound, LeftBound193347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority193351.actual selector witness, LeftBound193347.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193355

namespace LeftBound193359
def owner : Owner := ⟨.program ⟨257⟩, ⟨50083⟩⟩
def transferEvent : Nat := 193359
def frameStart : Nat := 193259
def rule : BoundRule := .sum [.predecessor 0 193357 .coefficient, .predecessor 1 193358 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193357 .coefficient)
      LeftBound193355.bound (LeftBound193355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193358 .coefficient)
      LeftBound193336.bound (LeftBound193336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193336.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound193355.bound, LeftBound193336.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193355.bound, LeftBound193336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound193355.actual selector witness, LeftBound193336.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193359

namespace LeftBound193372
def owner : Owner := ⟨.program ⟨257⟩, ⟨50082⟩⟩
def transferEvent : Nat := 193372
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 193370 .coefficient, .predecessor 1 193371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193370 .coefficient)
      LeftBound193201.bound (LeftBound193201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193371 .coefficient)
      LeftBound193184.bound (LeftBound193184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events754.exact193191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193184.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound193201.bound, LeftBound193184.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193201.bound, LeftBound193184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound193201.actual selector witness, LeftBound193184.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193372

namespace LeftBound193375
def owner : Owner := ⟨.program ⟨257⟩, ⟨50082⟩⟩
def transferEvent : Nat := 193375
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 193369 .summary, .result 193191 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 193369 .summary)
      LeftBound193203.bound (LeftBound193203.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48939⟩⟩) (rawTerms := some (Proof.Events755.exact193369RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound193203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 193191 .summary)
      LeftBound193186.bound (LeftBound193186.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50081⟩⟩) (rawTerms := some (Proof.Events754.exact193191RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound193186.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound193203.bound, LeftBound193186.bound]
def bound : CoeffClass := .finite ⟨32194504275408640829496428331008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193203.bound, LeftBound193186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound193203.actual selector witness, LeftBound193186.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193375

namespace LeftBound193399
def owner : Owner := ⟨.program ⟨257⟩, ⟨45205⟩⟩
def transferEvent : Nat := 193399
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 193397 .coefficient) (.predecessor 1 193398 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193397 .coefficient)
      LeftAuthority9092.bound (LeftAuthority9092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193398 .coefficient)
      LeftBound192901.bound (LeftBound192901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority9092.bound LeftBound192901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9092.bound, LeftBound192901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority9092.actual selector witness) * (LeftBound192901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound193399

namespace LeftBound193404
def owner : Owner := ⟨.program ⟨257⟩, ⟨8818⟩⟩
def transferEvent : Nat := 193404
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 193402 .coefficient) (.predecessor 1 193403 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193402 .coefficient)
      LeftBound192772.bound (LeftBound192772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193403 .coefficient)
      LeftBound17580.bound (LeftBound17580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17580.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound192772.bound LeftBound17580.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192772.bound, LeftBound17580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound192772.actual selector witness) * (LeftBound17580.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193404

namespace LeftBound193409
def owner : Owner := ⟨.program ⟨257⟩, ⟨45206⟩⟩
def transferEvent : Nat := 193409
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 193407 .coefficient, .predecessor 1 193408 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193407 .coefficient)
      LeftBound193404.bound (LeftBound193404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193404.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193404.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193408 .coefficient)
      LeftBound193399.bound (LeftBound193399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193399.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193399.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound193404.bound, LeftBound193399.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193404.bound, LeftBound193399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound193404.actual selector witness, LeftBound193399.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193409

namespace LeftBound193413
def owner : Owner := ⟨.program ⟨257⟩, ⟨45207⟩⟩
def transferEvent : Nat := 193413
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 193411 .coefficient, .predecessor 1 193412 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193411 .coefficient)
      LeftBound193409.bound (LeftBound193409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193412 .coefficient)
      LeftBound17572.bound (LeftBound17572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17572.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound193409.bound, LeftBound17572.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193409.bound, LeftBound17572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound193409.actual selector witness, LeftBound17572.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193413

namespace LeftBound193414
def owner : Owner := ⟨.program ⟨257⟩, ⟨45207⟩⟩
def transferEvent : Nat := 193414
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨110⟩⟩]⟩ [⟨.result 17573 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17573 .coefficient)
      LeftBound17572.bound (LeftBound17572.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨110⟩⟩) (rawTerms := some (Proof.Events068.exact17573RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17572.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound17572.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound17572.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound193414

namespace LeftBound193419
def owner : Owner := ⟨.program ⟨257⟩, ⟨45208⟩⟩
def transferEvent : Nat := 193419
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 193417 .coefficient) (.predecessor 1 193418 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193417 .coefficient)
      LeftBound193413.bound (LeftBound193413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193413.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193413.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193418 .coefficient)
      LeftAuthority9095.bound (LeftAuthority9095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9095.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound193413.bound LeftAuthority9095.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193413.bound, LeftAuthority9095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound193413.actual selector witness) * (LeftAuthority9095.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193419

namespace LeftBound193420
def owner : Owner := ⟨.program ⟨257⟩, ⟨45208⟩⟩
def transferEvent : Nat := 193420
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨14811⟩⟩], []⟩ [⟨.result 9096 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 9096 .coefficient)
      LeftAuthority9095.bound (LeftAuthority9095.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨14811⟩⟩) (rawTerms := some (Proof.Events035.exact9096RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9095.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9095.bound []
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority9095.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound193420

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
