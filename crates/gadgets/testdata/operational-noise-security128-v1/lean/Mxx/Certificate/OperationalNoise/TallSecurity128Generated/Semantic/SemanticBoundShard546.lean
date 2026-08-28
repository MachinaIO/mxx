import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard479

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84739
def owner : Owner := ⟨.program ⟨257⟩, ⟨68432⟩⟩
def transferEvent : Nat := 84739
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 84737 .coefficient) (.value (.predecessor 1 84738 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84737 .coefficient)
      LeftAuthority84735.bound (LeftAuthority84735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84738 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority84735.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84735.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority84735.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84739

namespace LeftBound84743
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def transferEvent : Nat := 84743
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84741 .coefficient) (.predecessor 1 84742 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84741 .coefficient)
      LeftBound75992.bound (LeftBound75992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84742 .coefficient)
      LeftBound84739.bound (LeftBound84739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84739.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75992.bound LeftBound84739.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75992.bound, LeftBound84739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75992.actual selector witness) * (LeftBound84739.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84743

namespace LeftBound84744
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def transferEvent : Nat := 84744
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩ [⟨.result 84736 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84736 .coefficient)
      LeftAuthority84735.bound (LeftAuthority84735.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68430⟩⟩) (rawTerms := some (Proof.Events331.exact84736RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84735.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority84735.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority84735.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84744

namespace LeftBound84745
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def transferEvent : Nat := 84745
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 75995 .summary) (.transfer 84744) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75995 .summary)
      LeftBound75993.bound (LeftBound75993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10368⟩⟩) (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 84744)
      LeftBound84744.bound (LeftBound84744.actual selector witness) := by
  exact .transfer (LeftBound84744.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75993.bound LeftBound84744.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75993.bound, LeftBound84744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75993.actual selector witness) * (LeftBound84744.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84745

namespace LeftBound85773
def owner : Owner := ⟨.program ⟨257⟩, ⟨18981⟩⟩
def transferEvent : Nat := 85773
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85771 .coefficient, .predecessor 1 85772 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85771 .coefficient)
      LeftAuthority85769.bound (LeftAuthority85769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85769.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85772 .coefficient)
      LeftAuthority85746.bound (LeftAuthority85746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85746.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85746.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority85769.bound, LeftAuthority85746.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85769.bound, LeftAuthority85746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority85769.actual selector witness, LeftAuthority85746.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85773

namespace LeftBound85777
def owner : Owner := ⟨.program ⟨257⟩, ⟨22201⟩⟩
def transferEvent : Nat := 85777
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85775 .coefficient, .predecessor 1 85776 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85775 .coefficient)
      LeftBound85773.bound (LeftBound85773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85776 .coefficient)
      LeftAuthority85723.bound (LeftAuthority85723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85723.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85723.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85773.bound, LeftAuthority85723.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85773.bound, LeftAuthority85723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85773.actual selector witness, LeftAuthority85723.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85777

namespace LeftBound85781
def owner : Owner := ⟨.program ⟨257⟩, ⟨32221⟩⟩
def transferEvent : Nat := 85781
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85779 .coefficient, .predecessor 1 85780 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85779 .coefficient)
      LeftBound85777.bound (LeftBound85777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85780 .coefficient)
      LeftAuthority85700.bound (LeftAuthority85700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85700.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85700.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85777.bound, LeftAuthority85700.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85777.bound, LeftAuthority85700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85777.actual selector witness, LeftAuthority85700.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85781

namespace LeftBound85785
def owner : Owner := ⟨.program ⟨257⟩, ⟨51276⟩⟩
def transferEvent : Nat := 85785
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85783 .coefficient, .predecessor 1 85784 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85783 .coefficient)
      LeftBound85781.bound (LeftBound85781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85784 .coefficient)
      LeftAuthority85677.bound (LeftAuthority85677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85677.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85781.bound, LeftAuthority85677.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85781.bound, LeftAuthority85677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85781.actual selector witness, LeftAuthority85677.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85785

namespace LeftBound85789
def owner : Owner := ⟨.program ⟨257⟩, ⟨54256⟩⟩
def transferEvent : Nat := 85789
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85787 .coefficient, .predecessor 1 85788 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85787 .coefficient)
      LeftBound85785.bound (LeftBound85785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85788 .coefficient)
      LeftAuthority85654.bound (LeftAuthority85654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85654.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85785.bound, LeftAuthority85654.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85785.bound, LeftAuthority85654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85785.actual selector witness, LeftAuthority85654.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85789

namespace LeftBound85793
def owner : Owner := ⟨.program ⟨257⟩, ⟨57236⟩⟩
def transferEvent : Nat := 85793
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85791 .coefficient, .predecessor 1 85792 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85791 .coefficient)
      LeftBound85789.bound (LeftBound85789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85792 .coefficient)
      LeftAuthority85631.bound (LeftAuthority85631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85789.bound, LeftAuthority85631.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85789.bound, LeftAuthority85631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85789.actual selector witness, LeftAuthority85631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85793

namespace LeftBound85797
def owner : Owner := ⟨.program ⟨257⟩, ⟨60216⟩⟩
def transferEvent : Nat := 85797
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85795 .coefficient, .predecessor 1 85796 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85795 .coefficient)
      LeftBound85793.bound (LeftBound85793.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85793.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85793.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85796 .coefficient)
      LeftAuthority85608.bound (LeftAuthority85608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85608.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85793.bound, LeftAuthority85608.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85793.bound, LeftAuthority85608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85793.actual selector witness, LeftAuthority85608.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85797

namespace LeftBound85801
def owner : Owner := ⟨.program ⟨257⟩, ⟨63196⟩⟩
def transferEvent : Nat := 85801
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85799 .coefficient, .predecessor 1 85800 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85799 .coefficient)
      LeftBound85797.bound (LeftBound85797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85800 .coefficient)
      LeftAuthority85585.bound (LeftAuthority85585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85797.bound, LeftAuthority85585.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85797.bound, LeftAuthority85585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85797.actual selector witness, LeftAuthority85585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85801

namespace LeftBound85805
def owner : Owner := ⟨.program ⟨257⟩, ⟨67022⟩⟩
def transferEvent : Nat := 85805
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85803 .coefficient, .predecessor 1 85804 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85803 .coefficient)
      LeftBound85801.bound (LeftBound85801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85804 .coefficient)
      LeftAuthority85562.bound (LeftAuthority85562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85562.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85801.bound, LeftAuthority85562.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85801.bound, LeftAuthority85562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85801.actual selector witness, LeftAuthority85562.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85805

namespace LeftBound85809
def owner : Owner := ⟨.program ⟨257⟩, ⟨67023⟩⟩
def transferEvent : Nat := 85809
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85807 .coefficient, .predecessor 1 85808 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85807 .coefficient)
      LeftBound85805.bound (LeftBound85805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85805.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85808 .coefficient)
      LeftAuthority85539.bound (LeftAuthority85539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85805.bound, LeftAuthority85539.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85805.bound, LeftAuthority85539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85805.actual selector witness, LeftAuthority85539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85809

namespace LeftBound85813
def owner : Owner := ⟨.program ⟨257⟩, ⟨67024⟩⟩
def transferEvent : Nat := 85813
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85811 .coefficient, .predecessor 1 85812 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85811 .coefficient)
      LeftBound85809.bound (LeftBound85809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85812 .coefficient)
      LeftAuthority85516.bound (LeftAuthority85516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85516.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85809.bound, LeftAuthority85516.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85809.bound, LeftAuthority85516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85809.actual selector witness, LeftAuthority85516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85813

namespace LeftBound85817
def owner : Owner := ⟨.program ⟨257⟩, ⟨67025⟩⟩
def transferEvent : Nat := 85817
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85815 .coefficient, .predecessor 1 85816 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85815 .coefficient)
      LeftBound85813.bound (LeftBound85813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85816 .coefficient)
      LeftAuthority85493.bound (LeftAuthority85493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85493.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85493.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85813.bound, LeftAuthority85493.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85813.bound, LeftAuthority85493.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85813.actual selector witness, LeftAuthority85493.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85817

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
