import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard082
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard475
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard478
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard492

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound77850
def owner : Owner := ⟨.program ⟨257⟩, ⟨10339⟩⟩
def transferEvent : Nat := 77850
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77848 .coefficient) (.predecessor 1 77849 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77848 .coefficient)
      LeftBound75772.bound (LeftBound75772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77849 .coefficient)
      LeftBound19083.bound (LeftBound19083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19083.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound75772.bound LeftBound19083.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75772.bound, LeftBound19083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound75772.actual selector witness) * (LeftBound19083.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77850

namespace LeftBound77855
def owner : Owner := ⟨.program ⟨257⟩, ⟨37262⟩⟩
def transferEvent : Nat := 77855
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77853 .coefficient, .predecessor 1 77854 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77853 .coefficient)
      LeftBound77850.bound (LeftBound77850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77854 .coefficient)
      LeftBound77845.bound (LeftBound77845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77850.bound, LeftBound77845.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77850.bound, LeftBound77845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77850.actual selector witness, LeftBound77845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77855

namespace LeftBound77859
def owner : Owner := ⟨.program ⟨257⟩, ⟨37263⟩⟩
def transferEvent : Nat := 77859
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77857 .coefficient, .predecessor 1 77858 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77857 .coefficient)
      LeftBound77855.bound (LeftBound77855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77858 .coefficient)
      LeftBound19075.bound (LeftBound19075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19075.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77855.bound, LeftBound19075.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77855.bound, LeftBound19075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77855.actual selector witness, LeftBound19075.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77859

namespace LeftBound77860
def owner : Owner := ⟨.program ⟨257⟩, ⟨37263⟩⟩
def transferEvent : Nat := 77860
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨107⟩⟩]⟩ [⟨.result 19076 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19076 .coefficient)
      LeftBound19075.bound (LeftBound19075.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨107⟩⟩) (rawTerms := some (Proof.Events074.exact19076RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19075.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19075.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19075.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77860

namespace LeftBound77865
def owner : Owner := ⟨.program ⟨257⟩, ⟨37264⟩⟩
def transferEvent : Nat := 77865
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77863 .coefficient) (.predecessor 1 77864 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77863 .coefficient)
      LeftBound77859.bound (LeftBound77859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77859.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77864 .coefficient)
      LeftAuthority3180.bound (LeftAuthority3180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3180.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound77859.bound LeftAuthority3180.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77859.bound, LeftAuthority3180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound77859.actual selector witness) * (LeftAuthority3180.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77865

namespace LeftBound77866
def owner : Owner := ⟨.program ⟨257⟩, ⟨37264⟩⟩
def transferEvent : Nat := 77866
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩], []⟩ [⟨.result 3181 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 3181 .coefficient)
      LeftAuthority3180.bound (LeftAuthority3180.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13971⟩⟩) (rawTerms := some (Proof.Events012.exact3181RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3180.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3180.bound []
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority3180.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77866

namespace LeftBound77867
def owner : Owner := ⟨.program ⟨257⟩, ⟨37264⟩⟩
def transferEvent : Nat := 77867
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 77862 .summary) (.transfer 77866) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 77862 .summary)
      LeftBound77860.bound (LeftBound77860.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37263⟩⟩) (rawTerms := some (Proof.Events304.exact77862RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 77866)
      LeftBound77866.bound (LeftBound77866.actual selector witness) := by
  exact .transfer (LeftBound77866.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound77860.bound LeftBound77866.bound
def bound : CoeffClass := .finite ⟨35782656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77860.bound, LeftBound77866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound77860.actual selector witness) * (LeftBound77866.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77867

namespace LeftBound77873
def owner : Owner := ⟨.program ⟨257⟩, ⟨13972⟩⟩
def transferEvent : Nat := 77873
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 77871 .coefficient) (.predecessor 1 77872 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77871 .coefficient)
      LeftAuthority3180.bound (LeftAuthority3180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77872 .coefficient)
      LeftBound75901.bound (LeftBound75901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority3180.bound LeftBound75901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3180.bound, LeftBound75901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority3180.actual selector witness) * (LeftBound75901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound77873

namespace LeftBound77878
def owner : Owner := ⟨.program ⟨257⟩, ⟨10356⟩⟩
def transferEvent : Nat := 77878
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77876 .coefficient) (.predecessor 1 77877 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77876 .coefficient)
      LeftBound75772.bound (LeftBound75772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77877 .coefficient)
      LeftBound19124.bound (LeftBound19124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound75772.bound LeftBound19124.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75772.bound, LeftBound19124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound75772.actual selector witness) * (LeftBound19124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77878

namespace LeftBound77883
def owner : Owner := ⟨.program ⟨257⟩, ⟨13973⟩⟩
def transferEvent : Nat := 77883
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77881 .coefficient, .predecessor 1 77882 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77881 .coefficient)
      LeftBound77878.bound (LeftBound77878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77878.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77882 .coefficient)
      LeftBound77873.bound (LeftBound77873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77878.bound, LeftBound77873.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77878.bound, LeftBound77873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77878.actual selector witness, LeftBound77873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77883

namespace LeftBound77887
def owner : Owner := ⟨.program ⟨257⟩, ⟨13974⟩⟩
def transferEvent : Nat := 77887
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77885 .coefficient, .predecessor 1 77886 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77885 .coefficient)
      LeftBound77883.bound (LeftBound77883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77886 .coefficient)
      LeftBound19116.bound (LeftBound19116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77883.bound, LeftBound19116.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77883.bound, LeftBound19116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77883.actual selector witness, LeftBound19116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77887

namespace LeftBound77888
def owner : Owner := ⟨.program ⟨257⟩, ⟨13974⟩⟩
def transferEvent : Nat := 77888
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨124⟩⟩]⟩ [⟨.result 19117 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19117 .coefficient)
      LeftBound19116.bound (LeftBound19116.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨124⟩⟩) (rawTerms := some (Proof.Events074.exact19117RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19116.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19116.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19116.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77888

namespace LeftBound77893
def owner : Owner := ⟨.program ⟨257⟩, ⟨13975⟩⟩
def transferEvent : Nat := 77893
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77891 .coefficient) (.predecessor 1 77892 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77891 .coefficient)
      LeftBound77887.bound (LeftBound77887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77892 .coefficient)
      LeftBound19113.bound (LeftBound19113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19113.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound77887.bound LeftBound19113.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77887.bound, LeftBound19113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound77887.actual selector witness) * (LeftBound19113.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77893

namespace LeftBound77894
def owner : Owner := ⟨.program ⟨257⟩, ⟨13975⟩⟩
def transferEvent : Nat := 77894
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩ [⟨.result 19110 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19110 .coefficient)
      LeftAuthority19109.bound (LeftAuthority19109.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9553⟩⟩) (rawTerms := some (Proof.Events074.exact19110RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19109.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19109.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19109.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority19109.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77894

namespace LeftBound77895
def owner : Owner := ⟨.program ⟨257⟩, ⟨13975⟩⟩
def transferEvent : Nat := 77895
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 77890 .summary) (.transfer 77894) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 77890 .summary)
      LeftBound77888.bound (LeftBound77888.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13974⟩⟩) (rawTerms := some (Proof.Events304.exact77890RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 77894)
      LeftBound77894.bound (LeftBound77894.actual selector witness) := by
  exact .transfer (LeftBound77894.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound77888.bound LeftBound77894.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77888.bound, LeftBound77894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound77888.actual selector witness) * (LeftBound77894.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77895

namespace LeftBound77903
def owner : Owner := ⟨.program ⟨257⟩, ⟨37265⟩⟩
def transferEvent : Nat := 77903
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77901 .coefficient, .predecessor 1 77902 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77901 .coefficient)
      LeftBound77893.bound (LeftBound77893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77902 .coefficient)
      LeftBound77865.bound (LeftBound77865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77865.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77865.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77893.bound, LeftBound77865.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77893.bound, LeftBound77865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77893.actual selector witness, LeftBound77865.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77903

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
