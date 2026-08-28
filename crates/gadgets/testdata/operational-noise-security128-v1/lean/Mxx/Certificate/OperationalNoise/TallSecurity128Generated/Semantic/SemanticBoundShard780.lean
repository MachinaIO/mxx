import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard779

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound119661
def owner : Owner := ⟨.program ⟨257⟩, ⟨9340⟩⟩
def transferEvent : Nat := 119661
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨18⟩⟩]⟩ [⟨.result 119611 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119611 .coefficient)
      LeftAuthority119610.bound (LeftAuthority119610.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨18⟩⟩) (rawTerms := some (Proof.Events467.exact119611RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority119610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority119610.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority119610.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority119610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority119610.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound119661

namespace LeftBound119666
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119666
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 119664 .coefficient) (.predecessor 1 119665 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119664 .coefficient)
      LeftBound119660.bound (LeftBound119660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119665 .coefficient)
      LeftBound6039.bound (LeftBound6039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound119660.bound LeftBound6039.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119660.bound, LeftBound6039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound119660.actual selector witness) * (LeftBound6039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound119666

namespace LeftBound119667
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119667
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], []⟩ [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 5815 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6774⟩⟩) (rawTerms := some (Proof.Events000.exact36RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5815 .coefficient)
      LeftAuthority5814.bound (LeftAuthority5814.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨67382⟩⟩) (rawTerms := some (Proof.Events022.exact5815RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5814.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35.bound [LeftAuthority5814.bound]
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35.bound, LeftAuthority5814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority35.actual selector witness) * ([LeftAuthority5814.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound119667

namespace LeftBound119668
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119668
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], []⟩ [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 5823 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 543 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6800⟩⟩) (rawTerms := some (Proof.Events002.exact543RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5823 .coefficient)
      LeftAuthority5822.bound (LeftAuthority5822.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨48307⟩⟩) (rawTerms := some (Proof.Events022.exact5823RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5822.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority542.bound [LeftAuthority5822.bound]
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority542.bound, LeftAuthority5822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority542.actual selector witness) * ([LeftAuthority5822.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound119668

namespace LeftBound119669
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119669
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 119667, .transfer 119668]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119667)
      LeftBound119667.bound (LeftBound119667.actual selector witness) := by
  exact .transfer (LeftBound119667.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119668)
      LeftBound119668.bound (LeftBound119668.actual selector witness) := by
  exact .transfer (LeftBound119668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119667.bound, LeftBound119668.bound]
def bound : CoeffClass := .finite ⟨4453112970957156472086120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119667.bound, LeftBound119668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119667.actual selector witness, LeftBound119668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119669

namespace LeftBound119670
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119670
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], []⟩ [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 5831 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 553 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6807⟩⟩) (rawTerms := some (Proof.Events002.exact553RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5831 .coefficient)
      LeftAuthority5830.bound (LeftAuthority5830.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨45627⟩⟩) (rawTerms := some (Proof.Events022.exact5831RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5830.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority552.bound [LeftAuthority5830.bound]
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority552.bound, LeftAuthority5830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority552.actual selector witness) * ([LeftAuthority5830.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound119670

namespace LeftBound119671
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119671
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 119669, .transfer 119670]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119669)
      LeftBound119669.bound (LeftBound119669.actual selector witness) := by
  exact .transfer (LeftBound119669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119670)
      LeftBound119670.bound (LeftBound119670.actual selector witness) := by
  exact .transfer (LeftBound119670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119669.bound, LeftBound119670.bound]
def bound : CoeffClass := .finite ⟨4683713856341753228595600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119669.bound, LeftBound119670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119669.actual selector witness, LeftBound119670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119671

namespace LeftBound119672
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119672
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩ [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 5839 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 563 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6817⟩⟩) (rawTerms := some (Proof.Events002.exact563RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5839 .coefficient)
      LeftAuthority5838.bound (LeftAuthority5838.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨42950⟩⟩) (rawTerms := some (Proof.Events022.exact5839RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5838.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5838.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority562.bound [LeftAuthority5838.bound]
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority562.bound, LeftAuthority5838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority562.actual selector witness) * ([LeftAuthority5838.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound119672

namespace LeftBound119673
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119673
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 119671, .transfer 119672]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119671)
      LeftBound119671.bound (LeftBound119671.actual selector witness) := by
  exact .transfer (LeftBound119671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119672)
      LeftBound119672.bound (LeftBound119672.actual selector witness) := by
  exact .transfer (LeftBound119672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119671.bound, LeftBound119672.bound]
def bound : CoeffClass := .finite ⟨4913864642405495209392960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119671.bound, LeftBound119672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119671.actual selector witness, LeftBound119672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119673

namespace LeftBound119674
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119674
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩ [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 5847 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 573 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6828⟩⟩) (rawTerms := some (Proof.Events002.exact573RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5847 .coefficient)
      LeftAuthority5846.bound (LeftAuthority5846.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨40270⟩⟩) (rawTerms := some (Proof.Events022.exact5847RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5846.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority572.bound [LeftAuthority5846.bound]
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority572.bound, LeftAuthority5846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority572.actual selector witness) * ([LeftAuthority5846.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound119674

namespace LeftBound119675
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119675
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 119673, .transfer 119674]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119673)
      LeftBound119673.bound (LeftBound119673.actual selector witness) := by
  exact .transfer (LeftBound119673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119674)
      LeftBound119674.bound (LeftBound119674.actual selector witness) := by
  exact .transfer (LeftBound119674.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119673.bound, LeftBound119674.bound]
def bound : CoeffClass := .finite ⟨5143450410172845024934680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119673.bound, LeftBound119674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119673.actual selector witness, LeftBound119674.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119675

namespace LeftBound119676
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119676
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩ [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 5855 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 583 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6838⟩⟩) (rawTerms := some (Proof.Events002.exact583RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5855 .coefficient)
      LeftAuthority5854.bound (LeftAuthority5854.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨37587⟩⟩) (rawTerms := some (Proof.Events022.exact5855RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5854.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority582.bound [LeftAuthority5854.bound]
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority582.bound, LeftAuthority5854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority582.actual selector witness) * ([LeftAuthority5854.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound119676

namespace LeftBound119677
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119677
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 119675, .transfer 119676]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119675)
      LeftBound119675.bound (LeftBound119675.actual selector witness) := by
  exact .transfer (LeftBound119675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119676)
      LeftBound119676.bound (LeftBound119676.actual selector witness) := by
  exact .transfer (LeftBound119676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119675.bound, LeftBound119676.bound]
def bound : CoeffClass := .finite ⟨5372571899340058642669440, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119675.bound, LeftBound119676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119675.actual selector witness, LeftBound119676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119677

namespace LeftBound119678
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119678
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩ [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 5863 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 593 .coefficient)
      LeftAuthority592.bound (LeftAuthority592.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6842⟩⟩) (rawTerms := some (Proof.Events002.exact593RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5863 .coefficient)
      LeftAuthority5862.bound (LeftAuthority5862.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨34907⟩⟩) (rawTerms := some (Proof.Events022.exact5863RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5862.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority592.bound [LeftAuthority5862.bound]
def bound : CoeffClass := .finite ⟨228855378262257504357600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority592.bound, LeftAuthority5862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority592.actual selector witness) * ([LeftAuthority5862.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound119678

namespace LeftBound119679
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119679
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 119677, .transfer 119678]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119677)
      LeftBound119677.bound (LeftBound119677.actual selector witness) := by
  exact .transfer (LeftBound119677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 119678)
      LeftBound119678.bound (LeftBound119678.actual selector witness) := by
  exact .transfer (LeftBound119678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119677.bound, LeftBound119678.bound]
def bound : CoeffClass := .finite ⟨5601427277602316147027040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119677.bound, LeftBound119678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119677.actual selector witness, LeftBound119678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119679

namespace LeftBound119680
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def transferEvent : Nat := 119680
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩ [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 5871 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 603 .coefficient)
      LeftAuthority602.bound (LeftAuthority602.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6857⟩⟩) (rawTerms := some (Proof.Events002.exact603RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5871 .coefficient)
      LeftAuthority5870.bound (LeftAuthority5870.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨29250⟩⟩) (rawTerms := some (Proof.Events022.exact5871RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5870.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority602.bound [LeftAuthority5870.bound]
def bound : CoeffClass := .finite ⟨228236850212900051643120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority602.bound, LeftAuthority5870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority602.actual selector witness) * ([LeftAuthority5870.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound119680

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
