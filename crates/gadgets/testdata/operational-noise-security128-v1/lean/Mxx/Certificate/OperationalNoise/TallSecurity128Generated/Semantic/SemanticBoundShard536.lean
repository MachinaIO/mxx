import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard130
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard475
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard478
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard535

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound83562
def owner : Owner := ⟨.program ⟨257⟩, ⟨23313⟩⟩
def transferEvent : Nat := 83562
def frameStart : Nat := 83489
def rule : BoundRule := .sum [.predecessor 0 83560 .coefficient, .predecessor 1 83561 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83560 .coefficient)
      LeftAuthority83558.bound (LeftAuthority83558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83561 .coefficient)
      LeftBound83554.bound (LeftBound83554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83554.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83554.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority83558.bound, LeftBound83554.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83558.bound, LeftBound83554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority83558.actual selector witness, LeftBound83554.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83562

namespace LeftBound83566
def owner : Owner := ⟨.program ⟨257⟩, ⟨24059⟩⟩
def transferEvent : Nat := 83566
def frameStart : Nat := 83489
def rule : BoundRule := .product (.predecessor 0 83564 .coefficient) (.predecessor 1 83565 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83564 .coefficient)
      LeftBound83562.bound (LeftBound83562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83562.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83565 .coefficient)
      LeftAuthority83539.bound (LeftAuthority83539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound83562.bound LeftAuthority83539.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83562.bound, LeftAuthority83539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound83562.actual selector witness) * (LeftAuthority83539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83566

namespace LeftBound83577
def owner : Owner := ⟨.program ⟨257⟩, ⟨22202⟩⟩
def transferEvent : Nat := 83577
def frameStart : Nat := 83489
def rule : BoundRule := .product (.predecessor 0 83575 .coefficient) (.predecessor 1 83576 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83575 .coefficient)
      LeftAuthority83550.bound (LeftAuthority83550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83576 .coefficient)
      LeftAuthority83573.bound (LeftAuthority83573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83573.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83573.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83550.bound LeftAuthority83573.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83550.bound, LeftAuthority83573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority83550.actual selector witness) * (LeftAuthority83573.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83577

namespace LeftBound83585
def owner : Owner := ⟨.program ⟨257⟩, ⟨22203⟩⟩
def transferEvent : Nat := 83585
def frameStart : Nat := 83489
def rule : BoundRule := .sum [.predecessor 0 83583 .coefficient, .predecessor 1 83584 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83583 .coefficient)
      LeftAuthority83581.bound (LeftAuthority83581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83581.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83584 .coefficient)
      LeftBound83577.bound (LeftBound83577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83577.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83577.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority83581.bound, LeftBound83577.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83581.bound, LeftBound83577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority83581.actual selector witness, LeftBound83577.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83585

namespace LeftBound83589
def owner : Owner := ⟨.program ⟨257⟩, ⟨24063⟩⟩
def transferEvent : Nat := 83589
def frameStart : Nat := 83489
def rule : BoundRule := .sum [.predecessor 0 83587 .coefficient, .predecessor 1 83588 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83587 .coefficient)
      LeftBound83585.bound (LeftBound83585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83588 .coefficient)
      LeftBound83566.bound (LeftBound83566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83585.bound, LeftBound83566.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83585.bound, LeftBound83566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83585.actual selector witness, LeftBound83566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83589

namespace LeftBound83602
def owner : Owner := ⟨.program ⟨257⟩, ⟨24061⟩⟩
def transferEvent : Nat := 83602
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83600 .coefficient, .predecessor 1 83601 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83600 .coefficient)
      LeftBound83431.bound (LeftBound83431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83601 .coefficient)
      LeftBound83414.bound (LeftBound83414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83431.bound, LeftBound83414.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83431.bound, LeftBound83414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83431.actual selector witness, LeftBound83414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83602

namespace LeftBound83605
def owner : Owner := ⟨.program ⟨257⟩, ⟨24061⟩⟩
def transferEvent : Nat := 83605
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 83599 .summary, .result 83421 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 83599 .summary)
      LeftBound83433.bound (LeftBound83433.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22799⟩⟩) (rawTerms := some (Proof.Events326.exact83599RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83433.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 83421 .summary)
      LeftBound83416.bound (LeftBound83416.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24060⟩⟩) (rawTerms := some (Proof.Events325.exact83421RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83433.bound, LeftBound83416.bound]
def bound : CoeffClass := .finite ⟨32189003662929394266751515230208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83433.bound, LeftBound83416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83433.actual selector witness, LeftBound83416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83605

namespace LeftBound83629
def owner : Owner := ⟨.program ⟨257⟩, ⟨18421⟩⟩
def transferEvent : Nat := 83629
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 83627 .coefficient) (.predecessor 1 83628 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83627 .coefficient)
      LeftAuthority3453.bound (LeftAuthority3453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3453.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3453.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83628 .coefficient)
      LeftBound75901.bound (LeftBound75901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority3453.bound LeftBound75901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3453.bound, LeftBound75901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority3453.actual selector witness) * (LeftBound75901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83629

namespace LeftBound83634
def owner : Owner := ⟨.program ⟨257⟩, ⟨10363⟩⟩
def transferEvent : Nat := 83634
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83632 .coefficient) (.predecessor 1 83633 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83632 .coefficient)
      LeftBound75772.bound (LeftBound75772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83633 .coefficient)
      LeftBound25095.bound (LeftBound25095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25095.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound75772.bound LeftBound25095.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75772.bound, LeftBound25095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound75772.actual selector witness) * (LeftBound25095.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83634

namespace LeftBound83639
def owner : Owner := ⟨.program ⟨257⟩, ⟨18422⟩⟩
def transferEvent : Nat := 83639
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83637 .coefficient, .predecessor 1 83638 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83637 .coefficient)
      LeftBound83634.bound (LeftBound83634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83634.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83638 .coefficient)
      LeftBound83629.bound (LeftBound83629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83629.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83634.bound, LeftBound83629.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83634.bound, LeftBound83629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83634.actual selector witness, LeftBound83629.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83639

namespace LeftBound83643
def owner : Owner := ⟨.program ⟨257⟩, ⟨18423⟩⟩
def transferEvent : Nat := 83643
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83641 .coefficient, .predecessor 1 83642 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83641 .coefficient)
      LeftBound83639.bound (LeftBound83639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83639.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83642 .coefficient)
      LeftBound25087.bound (LeftBound25087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83639.bound, LeftBound25087.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83639.bound, LeftBound25087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83639.actual selector witness, LeftBound25087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83643

namespace LeftBound83644
def owner : Owner := ⟨.program ⟨257⟩, ⟨18423⟩⟩
def transferEvent : Nat := 83644
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨131⟩⟩]⟩ [⟨.result 25088 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25088 .coefficient)
      LeftBound25087.bound (LeftBound25087.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨131⟩⟩) (rawTerms := some (Proof.Events098.exact25088RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25087.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25087.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25087.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83644

namespace LeftBound83649
def owner : Owner := ⟨.program ⟨257⟩, ⟨18424⟩⟩
def transferEvent : Nat := 83649
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83647 .coefficient) (.predecessor 1 83648 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83647 .coefficient)
      LeftBound83643.bound (LeftBound83643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83648 .coefficient)
      LeftAuthority3456.bound (LeftAuthority3456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound83643.bound LeftAuthority3456.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83643.bound, LeftAuthority3456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound83643.actual selector witness) * (LeftAuthority3456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83649

namespace LeftBound83650
def owner : Owner := ⟨.program ⟨257⟩, ⟨18424⟩⟩
def transferEvent : Nat := 83650
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩], []⟩ [⟨.result 3457 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 3457 .coefficient)
      LeftAuthority3456.bound (LeftAuthority3456.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨12771⟩⟩) (rawTerms := some (Proof.Events013.exact3457RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3456.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3456.bound []
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority3456.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83650

namespace LeftBound83651
def owner : Owner := ⟨.program ⟨257⟩, ⟨18424⟩⟩
def transferEvent : Nat := 83651
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 83646 .summary) (.transfer 83650) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 83646 .summary)
      LeftBound83644.bound (LeftBound83644.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18423⟩⟩) (rawTerms := some (Proof.Events326.exact83646RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 83650)
      LeftBound83650.bound (LeftBound83650.actual selector witness) := by
  exact .transfer (LeftBound83650.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound83644.bound LeftBound83650.bound
def bound : CoeffClass := .finite ⟨2555904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83644.bound, LeftBound83650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound83644.actual selector witness) * (LeftBound83650.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83651

namespace LeftBound83657
def owner : Owner := ⟨.program ⟨257⟩, ⟨12772⟩⟩
def transferEvent : Nat := 83657
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 83655 .coefficient) (.predecessor 1 83656 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83655 .coefficient)
      LeftAuthority3456.bound (LeftAuthority3456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83656 .coefficient)
      LeftBound75901.bound (LeftBound75901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority3456.bound LeftBound75901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3456.bound, LeftBound75901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority3456.actual selector witness) * (LeftBound75901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83657

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
