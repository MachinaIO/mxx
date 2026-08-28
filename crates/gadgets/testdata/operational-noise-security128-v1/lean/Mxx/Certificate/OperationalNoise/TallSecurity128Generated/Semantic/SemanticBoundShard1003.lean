import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard086
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard982
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard985
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard986
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1002

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound151256
def owner : Owner := ⟨.program ⟨257⟩, ⟨38119⟩⟩
def transferEvent : Nat := 151256
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 149120 .summary) (.transfer 151255) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 149120 .summary)
      LeftBound149118.bound (LeftBound149118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5545⟩⟩) (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound149118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 151255)
      LeftBound151255.bound (LeftBound151255.actual selector witness) := by
  exact .transfer (LeftBound151255.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149118.bound LeftBound151255.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149118.bound, LeftBound151255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149118.actual selector witness) * (LeftBound151255.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151256

namespace LeftBound151351
def owner : Owner := ⟨.program ⟨257⟩, ⟨37405⟩⟩
def transferEvent : Nat := 151351
def frameStart : Nat := 151312
def rule : BoundRule := .identity (.predecessor 0 151350 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151350 .coefficient)
      LeftAuthority151348.bound (LeftAuthority151348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151348.derived selector witness)

def rawBound : CoeffClass := LeftAuthority151348.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority151348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority151348.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound151351

namespace LeftBound151368
def owner : Owner := ⟨.program ⟨257⟩, ⟨38774⟩⟩
def transferEvent : Nat := 151368
def frameStart : Nat := 151312
def rule : BoundRule := .sum [.predecessor 0 151366 .coefficient, .predecessor 1 151367 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151366 .coefficient)
      LeftBound151351.bound (LeftBound151351.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound151351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151367 .coefficient)
      LeftAuthority151364.bound (LeftAuthority151364.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority151364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151351.bound, LeftAuthority151364.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151351.bound, LeftAuthority151364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151351.actual selector witness, LeftAuthority151364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151368

namespace LeftBound151371
def owner : Owner := ⟨.program ⟨257⟩, ⟨38775⟩⟩
def transferEvent : Nat := 151371
def frameStart : Nat := 151312
def rule : BoundRule := .identity (.predecessor 0 151370 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151370 .coefficient)
      LeftBound151368.bound (LeftBound151368.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound151368.derived selector witness)

def rawBound : CoeffClass := LeftBound151368.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound151368.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound151371

namespace LeftBound151377
def owner : Owner := ⟨.program ⟨257⟩, ⟨38776⟩⟩
def transferEvent : Nat := 151377
def frameStart : Nat := 151312
def rule : BoundRule := .product (.predecessor 0 151375 .coefficient) (.predecessor 1 151376 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151375 .coefficient)
      LeftAuthority151373.bound (LeftAuthority151373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151373.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151376 .coefficient)
      LeftBound151371.bound (LeftBound151371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151371.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority151373.bound LeftBound151371.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority151373.bound, LeftBound151371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority151373.actual selector witness) * (LeftBound151371.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151377

namespace LeftBound151385
def owner : Owner := ⟨.program ⟨257⟩, ⟨38777⟩⟩
def transferEvent : Nat := 151385
def frameStart : Nat := 151312
def rule : BoundRule := .sum [.predecessor 0 151383 .coefficient, .predecessor 1 151384 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151383 .coefficient)
      LeftAuthority151381.bound (LeftAuthority151381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151381.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151384 .coefficient)
      LeftBound151377.bound (LeftBound151377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151377.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151377.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority151381.bound, LeftBound151377.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority151381.bound, LeftBound151377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority151381.actual selector witness, LeftBound151377.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151385

namespace LeftBound151389
def owner : Owner := ⟨.program ⟨257⟩, ⟨39235⟩⟩
def transferEvent : Nat := 151389
def frameStart : Nat := 151312
def rule : BoundRule := .product (.predecessor 0 151387 .coefficient) (.predecessor 1 151388 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151387 .coefficient)
      LeftBound151385.bound (LeftBound151385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151388 .coefficient)
      LeftAuthority151362.bound (LeftAuthority151362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151362.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound151385.bound LeftAuthority151362.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151385.bound, LeftAuthority151362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound151385.actual selector witness) * (LeftAuthority151362.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151389

namespace LeftBound151400
def owner : Owner := ⟨.program ⟨257⟩, ⟨37605⟩⟩
def transferEvent : Nat := 151400
def frameStart : Nat := 151312
def rule : BoundRule := .product (.predecessor 0 151398 .coefficient) (.predecessor 1 151399 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151398 .coefficient)
      LeftAuthority151373.bound (LeftAuthority151373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151373.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151399 .coefficient)
      LeftAuthority151396.bound (LeftAuthority151396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151396.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151396.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority151373.bound LeftAuthority151396.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority151373.bound, LeftAuthority151396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority151373.actual selector witness) * (LeftAuthority151396.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151400

namespace LeftBound151408
def owner : Owner := ⟨.program ⟨257⟩, ⟨37606⟩⟩
def transferEvent : Nat := 151408
def frameStart : Nat := 151312
def rule : BoundRule := .sum [.predecessor 0 151406 .coefficient, .predecessor 1 151407 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151406 .coefficient)
      LeftAuthority151404.bound (LeftAuthority151404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151404.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151407 .coefficient)
      LeftBound151400.bound (LeftBound151400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151400.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority151404.bound, LeftBound151400.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority151404.bound, LeftBound151400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority151404.actual selector witness, LeftBound151400.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151408

namespace LeftBound151412
def owner : Owner := ⟨.program ⟨257⟩, ⟨39238⟩⟩
def transferEvent : Nat := 151412
def frameStart : Nat := 151312
def rule : BoundRule := .sum [.predecessor 0 151410 .coefficient, .predecessor 1 151411 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151410 .coefficient)
      LeftBound151408.bound (LeftBound151408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151411 .coefficient)
      LeftBound151389.bound (LeftBound151389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151408.bound, LeftBound151389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151408.bound, LeftBound151389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151408.actual selector witness, LeftBound151389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151412

namespace LeftBound151425
def owner : Owner := ⟨.program ⟨257⟩, ⟨39237⟩⟩
def transferEvent : Nat := 151425
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 151423 .coefficient, .predecessor 1 151424 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151423 .coefficient)
      LeftBound151254.bound (LeftBound151254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151254.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151424 .coefficient)
      LeftBound151237.bound (LeftBound151237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events590.exact151244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151254.bound, LeftBound151237.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151254.bound, LeftBound151237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151254.actual selector witness, LeftBound151237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151425

namespace LeftBound151428
def owner : Owner := ⟨.program ⟨257⟩, ⟨39237⟩⟩
def transferEvent : Nat := 151428
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 151422 .summary, .result 151244 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 151422 .summary)
      LeftBound151256.bound (LeftBound151256.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38119⟩⟩) (rawTerms := some (Proof.Events591.exact151422RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound151256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 151244 .summary)
      LeftBound151239.bound (LeftBound151239.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39236⟩⟩) (rawTerms := some (Proof.Events590.exact151244RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound151239.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151256.bound, LeftBound151239.bound]
def bound : CoeffClass := .finite ⟨32192736221397454434328420548608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151256.bound, LeftBound151239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151256.actual selector witness, LeftBound151239.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151428

namespace LeftBound151452
def owner : Owner := ⟨.program ⟨257⟩, ⟨34365⟩⟩
def transferEvent : Nat := 151452
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 151450 .coefficient) (.predecessor 1 151451 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151450 .coefficient)
      LeftAuthority6940.bound (LeftAuthority6940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6940.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151451 .coefficient)
      LeftBound149026.bound (LeftBound149026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority6940.bound LeftBound149026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6940.bound, LeftBound149026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority6940.actual selector witness) * (LeftBound149026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound151452

namespace LeftBound151457
def owner : Owner := ⟨.program ⟨257⟩, ⟨8244⟩⟩
def transferEvent : Nat := 151457
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 151455 .coefficient) (.predecessor 1 151456 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151455 .coefficient)
      LeftBound148897.bound (LeftBound148897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events581.exact148898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151456 .coefficient)
      LeftBound19584.bound (LeftBound19584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19584.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound148897.bound LeftBound19584.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148897.bound, LeftBound19584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound148897.actual selector witness) * (LeftBound19584.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151457

namespace LeftBound151462
def owner : Owner := ⟨.program ⟨257⟩, ⟨34366⟩⟩
def transferEvent : Nat := 151462
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 151460 .coefficient, .predecessor 1 151461 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151460 .coefficient)
      LeftBound151457.bound (LeftBound151457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151457.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151461 .coefficient)
      LeftBound151452.bound (LeftBound151452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151457.bound, LeftBound151452.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151457.bound, LeftBound151452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151457.actual selector witness, LeftBound151452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151462

namespace LeftBound151466
def owner : Owner := ⟨.program ⟨257⟩, ⟨34367⟩⟩
def transferEvent : Nat := 151466
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 151464 .coefficient, .predecessor 1 151465 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151464 .coefficient)
      LeftBound151462.bound (LeftBound151462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151465 .coefficient)
      LeftBound19576.bound (LeftBound19576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151462.bound, LeftBound19576.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151462.bound, LeftBound19576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151462.actual selector witness, LeftBound19576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151466

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
