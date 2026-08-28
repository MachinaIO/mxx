import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard102
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1794
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1796
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1798
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1828

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound270184
def owner : Owner := ⟨.program ⟨257⟩, ⟨67914⟩⟩
def transferEvent : Nat := 270184
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 266120 .summary) (.transfer 270183) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 266120 .summary)
      LeftBound266118.bound (LeftBound266118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5449⟩⟩) (rawTerms := some (Proof.Events1039.exact266120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound266118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 270183)
      LeftBound270183.bound (LeftBound270183.actual selector witness) := by
  exact .transfer (LeftBound270183.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound266118.bound LeftBound270183.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound266118.bound, LeftBound270183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound266118.actual selector witness) * (LeftBound270183.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound270184

namespace LeftBound270279
def owner : Owner := ⟨.program ⟨257⟩, ⟨65723⟩⟩
def transferEvent : Nat := 270279
def frameStart : Nat := 270240
def rule : BoundRule := .identity (.predecessor 0 270278 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270278 .coefficient)
      LeftAuthority270276.bound (LeftAuthority270276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority270276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority270276.derived selector witness)

def rawBound : CoeffClass := LeftAuthority270276.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority270276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority270276.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound270279

namespace LeftBound270296
def owner : Owner := ⟨.program ⟨257⟩, ⟨68975⟩⟩
def transferEvent : Nat := 270296
def frameStart : Nat := 270240
def rule : BoundRule := .sum [.predecessor 0 270294 .coefficient, .predecessor 1 270295 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270294 .coefficient)
      LeftBound270279.bound (LeftBound270279.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound270279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270295 .coefficient)
      LeftAuthority270292.bound (LeftAuthority270292.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority270292.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound270279.bound, LeftAuthority270292.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound270279.bound, LeftAuthority270292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound270279.actual selector witness, LeftAuthority270292.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound270296

namespace LeftBound270299
def owner : Owner := ⟨.program ⟨257⟩, ⟨68976⟩⟩
def transferEvent : Nat := 270299
def frameStart : Nat := 270240
def rule : BoundRule := .identity (.predecessor 0 270298 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270298 .coefficient)
      LeftBound270296.bound (LeftBound270296.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound270296.derived selector witness)

def rawBound : CoeffClass := LeftBound270296.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound270296.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound270296.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound270299

namespace LeftBound270305
def owner : Owner := ⟨.program ⟨257⟩, ⟨68977⟩⟩
def transferEvent : Nat := 270305
def frameStart : Nat := 270240
def rule : BoundRule := .product (.predecessor 0 270303 .coefficient) (.predecessor 1 270304 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270303 .coefficient)
      LeftAuthority270301.bound (LeftAuthority270301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority270301.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority270301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270304 .coefficient)
      LeftBound270299.bound (LeftBound270299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270299.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority270301.bound LeftBound270299.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority270301.bound, LeftBound270299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority270301.actual selector witness) * (LeftBound270299.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound270305

namespace LeftBound270313
def owner : Owner := ⟨.program ⟨257⟩, ⟨68978⟩⟩
def transferEvent : Nat := 270313
def frameStart : Nat := 270240
def rule : BoundRule := .sum [.predecessor 0 270311 .coefficient, .predecessor 1 270312 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270311 .coefficient)
      LeftAuthority270309.bound (LeftAuthority270309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority270309.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority270309.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270312 .coefficient)
      LeftBound270305.bound (LeftBound270305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270305.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270305.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority270309.bound, LeftBound270305.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority270309.bound, LeftBound270305.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority270309.actual selector witness, LeftBound270305.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound270313

namespace LeftBound270317
def owner : Owner := ⟨.program ⟨257⟩, ⟨69521⟩⟩
def transferEvent : Nat := 270317
def frameStart : Nat := 270240
def rule : BoundRule := .product (.predecessor 0 270315 .coefficient) (.predecessor 1 270316 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270315 .coefficient)
      LeftBound270313.bound (LeftBound270313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270316 .coefficient)
      LeftAuthority270290.bound (LeftAuthority270290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority270290.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority270290.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound270313.bound LeftAuthority270290.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound270313.bound, LeftAuthority270290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound270313.actual selector witness) * (LeftAuthority270290.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound270317

namespace LeftBound270328
def owner : Owner := ⟨.program ⟨257⟩, ⟨66030⟩⟩
def transferEvent : Nat := 270328
def frameStart : Nat := 270240
def rule : BoundRule := .product (.predecessor 0 270326 .coefficient) (.predecessor 1 270327 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270326 .coefficient)
      LeftAuthority270301.bound (LeftAuthority270301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority270301.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority270301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270327 .coefficient)
      LeftAuthority270324.bound (LeftAuthority270324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority270324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority270324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority270301.bound LeftAuthority270324.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority270301.bound, LeftAuthority270324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority270301.actual selector witness) * (LeftAuthority270324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound270328

namespace LeftBound270336
def owner : Owner := ⟨.program ⟨257⟩, ⟨66031⟩⟩
def transferEvent : Nat := 270336
def frameStart : Nat := 270240
def rule : BoundRule := .sum [.predecessor 0 270334 .coefficient, .predecessor 1 270335 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270334 .coefficient)
      LeftAuthority270332.bound (LeftAuthority270332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority270332.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority270332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270335 .coefficient)
      LeftBound270328.bound (LeftBound270328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270328.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority270332.bound, LeftBound270328.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority270332.bound, LeftBound270328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority270332.actual selector witness, LeftBound270328.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound270336

namespace LeftBound270340
def owner : Owner := ⟨.program ⟨257⟩, ⟨69533⟩⟩
def transferEvent : Nat := 270340
def frameStart : Nat := 270240
def rule : BoundRule := .sum [.predecessor 0 270338 .coefficient, .predecessor 1 270339 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270338 .coefficient)
      LeftBound270336.bound (LeftBound270336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1056.exact270337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270339 .coefficient)
      LeftBound270317.bound (LeftBound270317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270317.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound270336.bound, LeftBound270317.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound270336.bound, LeftBound270317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound270336.actual selector witness, LeftBound270317.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound270340

namespace LeftBound270353
def owner : Owner := ⟨.program ⟨257⟩, ⟨69523⟩⟩
def transferEvent : Nat := 270353
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 270351 .coefficient, .predecessor 1 270352 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270351 .coefficient)
      LeftBound270182.bound (LeftBound270182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1056.exact270350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270352 .coefficient)
      LeftBound270165.bound (LeftBound270165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1055.exact270172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270165.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound270182.bound, LeftBound270165.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound270182.bound, LeftBound270165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound270182.actual selector witness, LeftBound270165.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound270353

namespace LeftBound270356
def owner : Owner := ⟨.program ⟨257⟩, ⟨69523⟩⟩
def transferEvent : Nat := 270356
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 270350 .summary, .result 270172 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 270350 .summary)
      LeftBound270184.bound (LeftBound270184.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67914⟩⟩) (rawTerms := some (Proof.Events1056.exact270350RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound270184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 270172 .summary)
      LeftBound270167.bound (LeftBound270167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69522⟩⟩) (rawTerms := some (Proof.Events1055.exact270172RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound270167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound270184.bound, LeftBound270167.bound]
def bound : CoeffClass := .finite ⟨32191361068277642793642192273408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound270184.bound, LeftBound270167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound270184.actual selector witness, LeftBound270167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound270356

namespace LeftBound270380
def owner : Owner := ⟨.program ⟨257⟩, ⟨25391⟩⟩
def transferEvent : Nat := 270380
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 270378 .coefficient) (.predecessor 1 270379 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270378 .coefficient)
      LeftAuthority13016.bound (LeftAuthority13016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13016.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270379 .coefficient)
      LeftBound266026.bound (LeftBound266026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13016.bound LeftBound266026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13016.bound, LeftBound266026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13016.actual selector witness) * (LeftBound266026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound270380

namespace LeftBound270385
def owner : Owner := ⟨.program ⟨257⟩, ⟨7631⟩⟩
def transferEvent : Nat := 270385
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 270383 .coefficient) (.predecessor 1 270384 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270383 .coefficient)
      LeftBound265897.bound (LeftBound265897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270384 .coefficient)
      LeftBound21588.bound (LeftBound21588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound265897.bound LeftBound21588.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265897.bound, LeftBound21588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound265897.actual selector witness) * (LeftBound21588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound270385

namespace LeftBound270390
def owner : Owner := ⟨.program ⟨257⟩, ⟨25392⟩⟩
def transferEvent : Nat := 270390
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 270388 .coefficient, .predecessor 1 270389 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270388 .coefficient)
      LeftBound270385.bound (LeftBound270385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1056.exact270387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270389 .coefficient)
      LeftBound270380.bound (LeftBound270380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1056.exact270382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound270385.bound, LeftBound270380.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound270385.bound, LeftBound270380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound270385.actual selector witness, LeftBound270380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound270390

namespace LeftBound270394
def owner : Owner := ⟨.program ⟨257⟩, ⟨25393⟩⟩
def transferEvent : Nat := 270394
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 270392 .coefficient, .predecessor 1 270393 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 270392 .coefficient)
      LeftBound270390.bound (LeftBound270390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1056.exact270391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 270393 .coefficient)
      LeftBound21580.bound (LeftBound21580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound270390.bound, LeftBound21580.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound270390.bound, LeftBound21580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound270390.actual selector witness, LeftBound21580.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound270394

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
