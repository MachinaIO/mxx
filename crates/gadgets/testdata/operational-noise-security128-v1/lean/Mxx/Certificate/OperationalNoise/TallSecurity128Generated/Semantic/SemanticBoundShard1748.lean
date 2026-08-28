import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1747

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound258327
def owner : Owner := ⟨.program ⟨257⟩, ⟨31351⟩⟩
def transferEvent : Nat := 258327
def frameStart : Nat := 258298
def rule : BoundRule := .product (.predecessor 0 258325 .coefficient) (.predecessor 1 258326 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258325 .coefficient)
      LeftAuthority258323.bound (LeftAuthority258323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258323.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258326 .coefficient)
      LeftAuthority258320.bound (LeftAuthority258320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258320.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority258323.bound LeftAuthority258320.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258323.bound, LeftAuthority258320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority258323.actual selector witness) * (LeftAuthority258320.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258327

namespace LeftBound258331
def owner : Owner := ⟨.program ⟨257⟩, ⟨31352⟩⟩
def transferEvent : Nat := 258331
def frameStart : Nat := 258298
def rule : BoundRule := .identity (.predecessor 0 258330 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258330 .coefficient)
      LeftBound258327.bound (LeftBound258327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258327.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258327.derived selector witness)

def rawBound : CoeffClass := LeftBound258327.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound258327.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound258331

namespace LeftBound258348
def owner : Owner := ⟨.program ⟨257⟩, ⟨33206⟩⟩
def transferEvent : Nat := 258348
def frameStart : Nat := 258298
def rule : BoundRule := .sum [.predecessor 0 258346 .coefficient, .predecessor 1 258347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258346 .coefficient)
      LeftBound258331.bound (LeftBound258331.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound258331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258347 .coefficient)
      LeftAuthority258344.bound (LeftAuthority258344.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority258344.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound258331.bound, LeftAuthority258344.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258331.bound, LeftAuthority258344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound258331.actual selector witness, LeftAuthority258344.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound258348

namespace LeftBound258351
def owner : Owner := ⟨.program ⟨257⟩, ⟨33207⟩⟩
def transferEvent : Nat := 258351
def frameStart : Nat := 258298
def rule : BoundRule := .identity (.predecessor 0 258350 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258350 .coefficient)
      LeftBound258348.bound (LeftBound258348.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound258348.derived selector witness)

def rawBound : CoeffClass := LeftBound258348.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound258348.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound258351

namespace LeftBound258357
def owner : Owner := ⟨.program ⟨257⟩, ⟨33208⟩⟩
def transferEvent : Nat := 258357
def frameStart : Nat := 258298
def rule : BoundRule := .product (.predecessor 0 258355 .coefficient) (.predecessor 1 258356 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258355 .coefficient)
      LeftAuthority258353.bound (LeftAuthority258353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258353.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258356 .coefficient)
      LeftBound258351.bound (LeftBound258351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258351.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority258353.bound LeftBound258351.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258353.bound, LeftBound258351.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority258353.actual selector witness) * (LeftBound258351.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258357

namespace LeftBound258373
def owner : Owner := ⟨.program ⟨257⟩, ⟨9578⟩⟩
def transferEvent : Nat := 258373
def frameStart : Nat := 258298
def rule : BoundRule := .scale (.predecessor 0 258371 .coefficient) (.value (.predecessor 1 258372 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258371 .coefficient)
      LeftAuthority258369.bound (LeftAuthority258369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258369.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258372 .coefficient)
      LeftAuthority258360.bound (LeftAuthority258360.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority258360.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority258369.bound LeftAuthority258360.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258369.bound, LeftAuthority258360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority258369.actual selector witness) * (LeftAuthority258360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound258373

namespace LeftBound258376
def owner : Owner := ⟨.program ⟨257⟩, ⟨7287⟩⟩
def transferEvent : Nat := 258376
def frameStart : Nat := 258298
def rule : BoundRule := .identity (.predecessor 0 258375 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258375 .coefficient)
      LeftAuthority258363.bound (LeftAuthority258363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258363.derived selector witness)

def rawBound : CoeffClass := LeftAuthority258363.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority258363.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound258376

namespace LeftBound258380
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def transferEvent : Nat := 258380
def frameStart : Nat := 258298
def rule : BoundRule := .product (.predecessor 0 258378 .coefficient) (.predecessor 1 258379 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258378 .coefficient)
      LeftBound258376.bound (LeftBound258376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258379 .coefficient)
      LeftBound258373.bound (LeftBound258373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258373.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound258376.bound LeftBound258373.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258376.bound, LeftBound258373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound258376.actual selector witness) * (LeftBound258373.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258380

namespace LeftBound258385
def owner : Owner := ⟨.program ⟨257⟩, ⟨33209⟩⟩
def transferEvent : Nat := 258385
def frameStart : Nat := 258298
def rule : BoundRule := .sum [.predecessor 0 258383 .coefficient, .predecessor 1 258384 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258383 .coefficient)
      LeftBound258380.bound (LeftBound258380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258384 .coefficient)
      LeftBound258357.bound (LeftBound258357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258357.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258357.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound258380.bound, LeftBound258357.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258380.bound, LeftBound258357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound258380.actual selector witness, LeftBound258357.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound258385

namespace LeftBound258389
def owner : Owner := ⟨.program ⟨257⟩, ⟨33407⟩⟩
def transferEvent : Nat := 258389
def frameStart : Nat := 258298
def rule : BoundRule := .product (.predecessor 0 258387 .coefficient) (.predecessor 1 258388 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258387 .coefficient)
      LeftBound258385.bound (LeftBound258385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258388 .coefficient)
      LeftAuthority258342.bound (LeftAuthority258342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258342.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound258385.bound LeftAuthority258342.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258385.bound, LeftAuthority258342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound258385.actual selector witness) * (LeftAuthority258342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258389

namespace LeftBound258400
def owner : Owner := ⟨.program ⟨257⟩, ⟨31790⟩⟩
def transferEvent : Nat := 258400
def frameStart : Nat := 258298
def rule : BoundRule := .product (.predecessor 0 258398 .coefficient) (.predecessor 1 258399 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258398 .coefficient)
      LeftAuthority258353.bound (LeftAuthority258353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258353.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258399 .coefficient)
      LeftAuthority258396.bound (LeftAuthority258396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258396.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258396.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority258353.bound LeftAuthority258396.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258353.bound, LeftAuthority258396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority258353.actual selector witness) * (LeftAuthority258396.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258400

namespace LeftBound258408
def owner : Owner := ⟨.program ⟨257⟩, ⟨31791⟩⟩
def transferEvent : Nat := 258408
def frameStart : Nat := 258298
def rule : BoundRule := .sum [.predecessor 0 258406 .coefficient, .predecessor 1 258407 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258406 .coefficient)
      LeftAuthority258404.bound (LeftAuthority258404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258404.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258407 .coefficient)
      LeftBound258400.bound (LeftBound258400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258400.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority258404.bound, LeftBound258400.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258404.bound, LeftBound258400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority258404.actual selector witness, LeftBound258400.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound258408

namespace LeftBound258412
def owner : Owner := ⟨.program ⟨257⟩, ⟨33408⟩⟩
def transferEvent : Nat := 258412
def frameStart : Nat := 258298
def rule : BoundRule := .sum [.predecessor 0 258410 .coefficient, .predecessor 1 258411 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258410 .coefficient)
      LeftBound258408.bound (LeftBound258408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258411 .coefficient)
      LeftBound258389.bound (LeftBound258389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound258408.bound, LeftBound258389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258408.bound, LeftBound258389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound258408.actual selector witness, LeftBound258389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound258412

namespace LeftBound258425
def owner : Owner := ⟨.program ⟨257⟩, ⟨33406⟩⟩
def transferEvent : Nat := 258425
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 258423 .coefficient, .predecessor 1 258424 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258423 .coefficient)
      LeftBound258246.bound (LeftBound258246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258424 .coefficient)
      LeftBound258229.bound (LeftBound258229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1008.exact258236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258229.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound258246.bound, LeftBound258229.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258246.bound, LeftBound258229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound258246.actual selector witness, LeftBound258229.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound258425

namespace LeftBound258428
def owner : Owner := ⟨.program ⟨257⟩, ⟨33406⟩⟩
def transferEvent : Nat := 258428
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 258422 .summary, .result 258236 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258422 .summary)
      LeftBound258248.bound (LeftBound258248.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32342⟩⟩) (rawTerms := some (Proof.Events1009.exact258422RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound258248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258236 .summary)
      LeftBound258231.bound (LeftBound258231.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33405⟩⟩) (rawTerms := some (Proof.Events1008.exact258236RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound258231.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound258248.bound, LeftBound258231.bound]
def bound : CoeffClass := .finite ⟨2997852872440114577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258248.bound, LeftBound258231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound258248.actual selector witness, LeftBound258231.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound258428

namespace LeftBound258432
def owner : Owner := ⟨.program ⟨257⟩, ⟨33739⟩⟩
def transferEvent : Nat := 258432
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 258430 .coefficient) (.predecessor 1 258431 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258430 .coefficient)
      LeftBound258425.bound (LeftBound258425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1009.exact258429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258431 .coefficient)
      LeftAuthority258151.bound (LeftAuthority258151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1008.exact258152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258151.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound258425.bound LeftAuthority258151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258425.bound, LeftAuthority258151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound258425.actual selector witness) * (LeftAuthority258151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258432

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
