import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard221
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard224
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard228
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard232
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard235
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard238

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40655
def owner : Owner := ⟨.program ⟨257⟩, ⟨18014⟩⟩
def transferEvent : Nat := 40655
def frameStart : Nat := 40578
def rule : BoundRule := .product (.predecessor 0 40653 .coefficient) (.predecessor 1 40654 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40653 .coefficient)
      LeftBound40651.bound (LeftBound40651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40651.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40654 .coefficient)
      LeftAuthority40628.bound (LeftAuthority40628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40628.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40628.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound40651.bound LeftAuthority40628.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40651.bound, LeftAuthority40628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound40651.actual selector witness) * (LeftAuthority40628.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40655

namespace LeftBound40666
def owner : Owner := ⟨.program ⟨257⟩, ⟨16180⟩⟩
def transferEvent : Nat := 40666
def frameStart : Nat := 40578
def rule : BoundRule := .product (.predecessor 0 40664 .coefficient) (.predecessor 1 40665 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40664 .coefficient)
      LeftAuthority40639.bound (LeftAuthority40639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40639.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40639.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40665 .coefficient)
      LeftAuthority40662.bound (LeftAuthority40662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40662.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority40639.bound LeftAuthority40662.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40639.bound, LeftAuthority40662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority40639.actual selector witness) * (LeftAuthority40662.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40666

namespace LeftBound40674
def owner : Owner := ⟨.program ⟨257⟩, ⟨16181⟩⟩
def transferEvent : Nat := 40674
def frameStart : Nat := 40578
def rule : BoundRule := .sum [.predecessor 0 40672 .coefficient, .predecessor 1 40673 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40672 .coefficient)
      LeftAuthority40670.bound (LeftAuthority40670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40670.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40670.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40673 .coefficient)
      LeftBound40666.bound (LeftBound40666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority40670.bound, LeftBound40666.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40670.bound, LeftBound40666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority40670.actual selector witness, LeftBound40666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40674

namespace LeftBound40678
def owner : Owner := ⟨.program ⟨257⟩, ⟨18017⟩⟩
def transferEvent : Nat := 40678
def frameStart : Nat := 40578
def rule : BoundRule := .sum [.predecessor 0 40676 .coefficient, .predecessor 1 40677 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40676 .coefficient)
      LeftBound40674.bound (LeftBound40674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40674.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40677 .coefficient)
      LeftBound40655.bound (LeftBound40655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40655.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40655.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40674.bound, LeftBound40655.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40674.bound, LeftBound40655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40674.actual selector witness, LeftBound40655.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40678

namespace LeftBound40691
def owner : Owner := ⟨.program ⟨257⟩, ⟨18016⟩⟩
def transferEvent : Nat := 40691
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40689 .coefficient, .predecessor 1 40690 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40689 .coefficient)
      LeftBound40520.bound (LeftBound40520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40690 .coefficient)
      LeftBound40503.bound (LeftBound40503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40503.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40520.bound, LeftBound40503.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40520.bound, LeftBound40503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40520.actual selector witness, LeftBound40503.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40691

namespace LeftBound40694
def owner : Owner := ⟨.program ⟨257⟩, ⟨18016⟩⟩
def transferEvent : Nat := 40694
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40688 .summary, .result 40510 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40688 .summary)
      LeftBound40522.bound (LeftBound40522.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16779⟩⟩) (rawTerms := some (Proof.Events158.exact40688RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40510 .summary)
      LeftBound40505.bound (LeftBound40505.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18015⟩⟩) (rawTerms := some (Proof.Events158.exact40510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40522.bound, LeftBound40505.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40522.bound, LeftBound40505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40522.actual selector witness, LeftBound40505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40694

namespace LeftBound40698
def owner : Owner := ⟨.program ⟨257⟩, ⟨20935⟩⟩
def transferEvent : Nat := 40698
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40696 .coefficient, .predecessor 1 40697 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40696 .coefficient)
      LeftBound40691.bound (LeftBound40691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40697 .coefficient)
      LeftBound40209.bound (LeftBound40209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40209.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40691.bound, LeftBound40209.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40691.bound, LeftBound40209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40691.actual selector witness, LeftBound40209.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40698

namespace LeftBound40699
def owner : Owner := ⟨.program ⟨257⟩, ⟨20935⟩⟩
def transferEvent : Nat := 40699
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40695 .summary, .result 40213 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40695 .summary)
      LeftBound40694.bound (LeftBound40694.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18016⟩⟩) (rawTerms := some (Proof.Events158.exact40695RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40213 .summary)
      LeftBound40212.bound (LeftBound40212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20934⟩⟩) (rawTerms := some (Proof.Events157.exact40213RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40212.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40694.bound, LeftBound40212.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40694.bound, LeftBound40212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40694.actual selector witness, LeftBound40212.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40699

namespace LeftBound40703
def owner : Owner := ⟨.program ⟨257⟩, ⟨24155⟩⟩
def transferEvent : Nat := 40703
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40701 .coefficient, .predecessor 1 40702 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40701 .coefficient)
      LeftBound40698.bound (LeftBound40698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40702 .coefficient)
      LeftBound39727.bound (LeftBound39727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40698.bound, LeftBound39727.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40698.bound, LeftBound39727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40698.actual selector witness, LeftBound39727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40703

namespace LeftBound40704
def owner : Owner := ⟨.program ⟨257⟩, ⟨24155⟩⟩
def transferEvent : Nat := 40704
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40700 .summary, .result 39731 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40700 .summary)
      LeftBound40699.bound (LeftBound40699.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20935⟩⟩) (rawTerms := some (Proof.Events158.exact40700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 39731 .summary)
      LeftBound39730.bound (LeftBound39730.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24154⟩⟩) (rawTerms := some (Proof.Events155.exact39731RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39730.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40699.bound, LeftBound39730.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40699.bound, LeftBound39730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40699.actual selector witness, LeftBound39730.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40704

namespace LeftBound40708
def owner : Owner := ⟨.program ⟨257⟩, ⟨34175⟩⟩
def transferEvent : Nat := 40708
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40706 .coefficient, .predecessor 1 40707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40706 .coefficient)
      LeftBound40703.bound (LeftBound40703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40707 .coefficient)
      LeftBound39245.bound (LeftBound39245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39245.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40703.bound, LeftBound39245.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40703.bound, LeftBound39245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40703.actual selector witness, LeftBound39245.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40708

namespace LeftBound40709
def owner : Owner := ⟨.program ⟨257⟩, ⟨34175⟩⟩
def transferEvent : Nat := 40709
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40705 .summary, .result 39249 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40705 .summary)
      LeftBound40704.bound (LeftBound40704.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24155⟩⟩) (rawTerms := some (Proof.Events159.exact40705RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 39249 .summary)
      LeftBound39248.bound (LeftBound39248.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34174⟩⟩) (rawTerms := some (Proof.Events153.exact39249RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39248.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40704.bound, LeftBound39248.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40704.bound, LeftBound39248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40704.actual selector witness, LeftBound39248.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40709

namespace LeftBound40713
def owner : Owner := ⟨.program ⟨257⟩, ⟨53235⟩⟩
def transferEvent : Nat := 40713
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40711 .coefficient, .predecessor 1 40712 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40711 .coefficient)
      LeftBound40708.bound (LeftBound40708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40712 .coefficient)
      LeftBound38763.bound (LeftBound38763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38763.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40708.bound, LeftBound38763.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40708.bound, LeftBound38763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40708.actual selector witness, LeftBound38763.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40713

namespace LeftBound40714
def owner : Owner := ⟨.program ⟨257⟩, ⟨53235⟩⟩
def transferEvent : Nat := 40714
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40710 .summary, .result 38767 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40710 .summary)
      LeftBound40709.bound (LeftBound40709.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34175⟩⟩) (rawTerms := some (Proof.Events159.exact40710RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 38767 .summary)
      LeftBound38766.bound (LeftBound38766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53234⟩⟩) (rawTerms := some (Proof.Events151.exact38767RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40709.bound, LeftBound38766.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40709.bound, LeftBound38766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40709.actual selector witness, LeftBound38766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40714

namespace LeftBound40718
def owner : Owner := ⟨.program ⟨257⟩, ⟨56215⟩⟩
def transferEvent : Nat := 40718
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40716 .coefficient, .predecessor 1 40717 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40716 .coefficient)
      LeftBound40713.bound (LeftBound40713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40713.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40717 .coefficient)
      LeftBound38281.bound (LeftBound38281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40713.bound, LeftBound38281.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40713.bound, LeftBound38281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40713.actual selector witness, LeftBound38281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40718

namespace LeftBound40719
def owner : Owner := ⟨.program ⟨257⟩, ⟨56215⟩⟩
def transferEvent : Nat := 40719
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40715 .summary, .result 38285 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40715 .summary)
      LeftBound40714.bound (LeftBound40714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53235⟩⟩) (rawTerms := some (Proof.Events159.exact40715RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 38285 .summary)
      LeftBound38284.bound (LeftBound38284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56214⟩⟩) (rawTerms := some (Proof.Events149.exact38285RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38284.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40714.bound, LeftBound38284.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40714.bound, LeftBound38284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40714.actual selector witness, LeftBound38284.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40719

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
