import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1998
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2057
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2088

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound307532
def owner : Owner := ⟨.program ⟨257⟩, ⟨20337⟩⟩
def transferEvent : Nat := 307532
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 302301 .summary) (.transfer 307531) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302301 .summary)
      LeftBound302300.bound (LeftBound302300.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20111⟩⟩) (rawTerms := some (Proof.Events1180.exact302301RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 307531)
      LeftBound307531.bound (LeftBound307531.actual selector witness) := by
  exact .transfer (LeftBound307531.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302300.bound LeftBound307531.bound
def bound : CoeffClass := .finite ⟨32188905437706348505289216491520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302300.bound, LeftBound307531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302300.actual selector witness) * (LeftBound307531.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307532

namespace LeftBound307543
def owner : Owner := ⟨.program ⟨257⟩, ⟨19254⟩⟩
def transferEvent : Nat := 307543
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 307541 .coefficient) (.value (.predecessor 1 307542 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307541 .coefficient)
      LeftAuthority307539.bound (LeftAuthority307539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307542 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority307539.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307539.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority307539.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound307543

namespace LeftBound307547
def owner : Owner := ⟨.program ⟨257⟩, ⟨19255⟩⟩
def transferEvent : Nat := 307547
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 307545 .coefficient) (.predecessor 1 307546 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307545 .coefficient)
      LeftBound295192.bound (LeftBound295192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307546 .coefficient)
      LeftBound307543.bound (LeftBound307543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307543.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295192.bound LeftBound307543.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295192.bound, LeftBound307543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295192.actual selector witness) * (LeftBound307543.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307547

namespace LeftBound307548
def owner : Owner := ⟨.program ⟨257⟩, ⟨19255⟩⟩
def transferEvent : Nat := 307548
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨19252⟩⟩]⟩ [⟨.result 307540 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307540 .coefficient)
      LeftAuthority307539.bound (LeftAuthority307539.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨19252⟩⟩) (rawTerms := some (Proof.Events1201.exact307540RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307539.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority307539.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority307539.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound307548

namespace LeftBound307549
def owner : Owner := ⟨.program ⟨257⟩, ⟨19255⟩⟩
def transferEvent : Nat := 307549
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 295195 .summary) (.transfer 307548) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295195 .summary)
      LeftBound295193.bound (LeftBound295193.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨2380⟩⟩) (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 307548)
      LeftBound307548.bound (LeftBound307548.actual selector witness) := by
  exact .transfer (LeftBound307548.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295193.bound LeftBound307548.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295193.bound, LeftBound307548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295193.actual selector witness) * (LeftBound307548.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307549

namespace LeftBound307620
def owner : Owner := ⟨.program ⟨257⟩, ⟨18509⟩⟩
def transferEvent : Nat := 307620
def frameStart : Nat := 307593
def rule : BoundRule := .identity (.predecessor 0 307619 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307619 .coefficient)
      LeftAuthority307617.bound (LeftAuthority307617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307617.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307617.derived selector witness)

def rawBound : CoeffClass := LeftAuthority307617.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority307617.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound307620

namespace LeftBound307637
def owner : Owner := ⟨.program ⟨257⟩, ⟨20026⟩⟩
def transferEvent : Nat := 307637
def frameStart : Nat := 307593
def rule : BoundRule := .sum [.predecessor 0 307635 .coefficient, .predecessor 1 307636 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307635 .coefficient)
      LeftBound307620.bound (LeftBound307620.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound307620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307636 .coefficient)
      LeftAuthority307633.bound (LeftAuthority307633.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority307633.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307620.bound, LeftAuthority307633.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307620.bound, LeftAuthority307633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307620.actual selector witness, LeftAuthority307633.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307637

namespace LeftBound307640
def owner : Owner := ⟨.program ⟨257⟩, ⟨20027⟩⟩
def transferEvent : Nat := 307640
def frameStart : Nat := 307593
def rule : BoundRule := .identity (.predecessor 0 307639 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307639 .coefficient)
      LeftBound307637.bound (LeftBound307637.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound307637.derived selector witness)

def rawBound : CoeffClass := LeftBound307637.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound307637.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound307640

namespace LeftBound307646
def owner : Owner := ⟨.program ⟨257⟩, ⟨20028⟩⟩
def transferEvent : Nat := 307646
def frameStart : Nat := 307593
def rule : BoundRule := .product (.predecessor 0 307644 .coefficient) (.predecessor 1 307645 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307644 .coefficient)
      LeftAuthority307642.bound (LeftAuthority307642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307645 .coefficient)
      LeftBound307640.bound (LeftBound307640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307640.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307640.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority307642.bound LeftBound307640.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307642.bound, LeftBound307640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority307642.actual selector witness) * (LeftBound307640.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307646

namespace LeftBound307654
def owner : Owner := ⟨.program ⟨257⟩, ⟨20029⟩⟩
def transferEvent : Nat := 307654
def frameStart : Nat := 307593
def rule : BoundRule := .sum [.predecessor 0 307652 .coefficient, .predecessor 1 307653 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307652 .coefficient)
      LeftAuthority307650.bound (LeftAuthority307650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307650.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307653 .coefficient)
      LeftBound307646.bound (LeftBound307646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307646.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307646.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority307650.bound, LeftBound307646.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307650.bound, LeftBound307646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority307650.actual selector witness, LeftBound307646.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307654

namespace LeftBound307658
def owner : Owner := ⟨.program ⟨257⟩, ⟨20336⟩⟩
def transferEvent : Nat := 307658
def frameStart : Nat := 307593
def rule : BoundRule := .product (.predecessor 0 307656 .coefficient) (.predecessor 1 307657 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307656 .coefficient)
      LeftBound307654.bound (LeftBound307654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307657 .coefficient)
      LeftAuthority307631.bound (LeftAuthority307631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307631.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound307654.bound LeftAuthority307631.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307654.bound, LeftAuthority307631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound307654.actual selector witness) * (LeftAuthority307631.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307658

namespace LeftBound307669
def owner : Owner := ⟨.program ⟨257⟩, ⟨18674⟩⟩
def transferEvent : Nat := 307669
def frameStart : Nat := 307593
def rule : BoundRule := .product (.predecessor 0 307667 .coefficient) (.predecessor 1 307668 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307667 .coefficient)
      LeftAuthority307642.bound (LeftAuthority307642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307668 .coefficient)
      LeftAuthority307665.bound (LeftAuthority307665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307665.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307665.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority307642.bound LeftAuthority307665.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307642.bound, LeftAuthority307665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority307642.actual selector witness) * (LeftAuthority307665.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307669

namespace LeftBound307677
def owner : Owner := ⟨.program ⟨257⟩, ⟨18675⟩⟩
def transferEvent : Nat := 307677
def frameStart : Nat := 307593
def rule : BoundRule := .sum [.predecessor 0 307675 .coefficient, .predecessor 1 307676 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307675 .coefficient)
      LeftAuthority307673.bound (LeftAuthority307673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307676 .coefficient)
      LeftBound307669.bound (LeftBound307669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307669.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority307673.bound, LeftBound307669.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307673.bound, LeftBound307669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority307673.actual selector witness, LeftBound307669.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307677

namespace LeftBound307681
def owner : Owner := ⟨.program ⟨257⟩, ⟨20341⟩⟩
def transferEvent : Nat := 307681
def frameStart : Nat := 307593
def rule : BoundRule := .sum [.predecessor 0 307679 .coefficient, .predecessor 1 307680 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307679 .coefficient)
      LeftBound307677.bound (LeftBound307677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307677.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307680 .coefficient)
      LeftBound307658.bound (LeftBound307658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307677.bound, LeftBound307658.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307677.bound, LeftBound307658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307677.actual selector witness, LeftBound307658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307681

namespace LeftBound307694
def owner : Owner := ⟨.program ⟨257⟩, ⟨20338⟩⟩
def transferEvent : Nat := 307694
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307692 .coefficient, .predecessor 1 307693 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307692 .coefficient)
      LeftBound307547.bound (LeftBound307547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307693 .coefficient)
      LeftBound307530.bound (LeftBound307530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307530.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307530.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307547.bound, LeftBound307530.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307547.bound, LeftBound307530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307547.actual selector witness, LeftBound307530.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307694

namespace LeftBound307697
def owner : Owner := ⟨.program ⟨257⟩, ⟨20338⟩⟩
def transferEvent : Nat := 307697
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307691 .summary, .result 307537 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307691 .summary)
      LeftBound307549.bound (LeftBound307549.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19255⟩⟩) (rawTerms := some (Proof.Events1201.exact307691RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307537 .summary)
      LeftBound307532.bound (LeftBound307532.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20337⟩⟩) (rawTerms := some (Proof.Events1201.exact307537RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307549.bound, LeftBound307532.bound]
def bound : CoeffClass := .finite ⟨32188905437706550578131070353408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307549.bound, LeftBound307532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307549.actual selector witness, LeftBound307532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307697

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
