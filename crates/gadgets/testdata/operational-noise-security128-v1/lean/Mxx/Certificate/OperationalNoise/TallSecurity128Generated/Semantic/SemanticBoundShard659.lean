import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard607
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard658

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound102221
def owner : Owner := ⟨.program ⟨257⟩, ⟨29129⟩⟩
def transferEvent : Nat := 102221
def frameStart : Nat := 102182
def rule : BoundRule := .identity (.predecessor 0 102220 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102220 .coefficient)
      LeftAuthority102218.bound (LeftAuthority102218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102218.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102218.derived selector witness)

def rawBound : CoeffClass := LeftAuthority102218.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority102218.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound102221

namespace LeftBound102238
def owner : Owner := ⟨.program ⟨257⟩, ⟨30466⟩⟩
def transferEvent : Nat := 102238
def frameStart : Nat := 102182
def rule : BoundRule := .sum [.predecessor 0 102236 .coefficient, .predecessor 1 102237 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102236 .coefficient)
      LeftBound102221.bound (LeftBound102221.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound102221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102237 .coefficient)
      LeftAuthority102234.bound (LeftAuthority102234.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority102234.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102221.bound, LeftAuthority102234.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102221.bound, LeftAuthority102234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound102221.actual selector witness, LeftAuthority102234.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102238

namespace LeftBound102241
def owner : Owner := ⟨.program ⟨257⟩, ⟨30467⟩⟩
def transferEvent : Nat := 102241
def frameStart : Nat := 102182
def rule : BoundRule := .identity (.predecessor 0 102240 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102240 .coefficient)
      LeftBound102238.bound (LeftBound102238.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound102238.derived selector witness)

def rawBound : CoeffClass := LeftBound102238.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound102238.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound102241

namespace LeftBound102247
def owner : Owner := ⟨.program ⟨257⟩, ⟨30468⟩⟩
def transferEvent : Nat := 102247
def frameStart : Nat := 102182
def rule : BoundRule := .product (.predecessor 0 102245 .coefficient) (.predecessor 1 102246 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102245 .coefficient)
      LeftAuthority102243.bound (LeftAuthority102243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102246 .coefficient)
      LeftBound102241.bound (LeftBound102241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102241.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority102243.bound LeftBound102241.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102243.bound, LeftBound102241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority102243.actual selector witness) * (LeftBound102241.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102247

namespace LeftBound102255
def owner : Owner := ⟨.program ⟨257⟩, ⟨30469⟩⟩
def transferEvent : Nat := 102255
def frameStart : Nat := 102182
def rule : BoundRule := .sum [.predecessor 0 102253 .coefficient, .predecessor 1 102254 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102253 .coefficient)
      LeftAuthority102251.bound (LeftAuthority102251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102251.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102254 .coefficient)
      LeftBound102247.bound (LeftBound102247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102247.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority102251.bound, LeftBound102247.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102251.bound, LeftBound102247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority102251.actual selector witness, LeftBound102247.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102255

namespace LeftBound102259
def owner : Owner := ⟨.program ⟨257⟩, ⟨31089⟩⟩
def transferEvent : Nat := 102259
def frameStart : Nat := 102182
def rule : BoundRule := .product (.predecessor 0 102257 .coefficient) (.predecessor 1 102258 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102257 .coefficient)
      LeftBound102255.bound (LeftBound102255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102258 .coefficient)
      LeftAuthority102232.bound (LeftAuthority102232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102232.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102232.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound102255.bound LeftAuthority102232.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102255.bound, LeftAuthority102232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound102255.actual selector witness) * (LeftAuthority102232.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102259

namespace LeftBound102270
def owner : Owner := ⟨.program ⟨257⟩, ⟨29369⟩⟩
def transferEvent : Nat := 102270
def frameStart : Nat := 102182
def rule : BoundRule := .product (.predecessor 0 102268 .coefficient) (.predecessor 1 102269 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102268 .coefficient)
      LeftAuthority102243.bound (LeftAuthority102243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102269 .coefficient)
      LeftAuthority102266.bound (LeftAuthority102266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority102243.bound LeftAuthority102266.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102243.bound, LeftAuthority102266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority102243.actual selector witness) * (LeftAuthority102266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102270

namespace LeftBound102278
def owner : Owner := ⟨.program ⟨257⟩, ⟨29370⟩⟩
def transferEvent : Nat := 102278
def frameStart : Nat := 102182
def rule : BoundRule := .sum [.predecessor 0 102276 .coefficient, .predecessor 1 102277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102276 .coefficient)
      LeftAuthority102274.bound (LeftAuthority102274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102274.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102277 .coefficient)
      LeftBound102270.bound (LeftBound102270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102270.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority102274.bound, LeftBound102270.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102274.bound, LeftBound102270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority102274.actual selector witness, LeftBound102270.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102278

namespace LeftBound102282
def owner : Owner := ⟨.program ⟨257⟩, ⟨31093⟩⟩
def transferEvent : Nat := 102282
def frameStart : Nat := 102182
def rule : BoundRule := .sum [.predecessor 0 102280 .coefficient, .predecessor 1 102281 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102280 .coefficient)
      LeftBound102278.bound (LeftBound102278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102281 .coefficient)
      LeftBound102259.bound (LeftBound102259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102259.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102278.bound, LeftBound102259.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102278.bound, LeftBound102259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound102278.actual selector witness, LeftBound102259.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102282

namespace LeftBound102295
def owner : Owner := ⟨.program ⟨257⟩, ⟨31091⟩⟩
def transferEvent : Nat := 102295
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102293 .coefficient, .predecessor 1 102294 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102293 .coefficient)
      LeftBound102124.bound (LeftBound102124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102294 .coefficient)
      LeftBound102107.bound (LeftBound102107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102107.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102124.bound, LeftBound102107.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102124.bound, LeftBound102107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound102124.actual selector witness, LeftBound102107.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102295

namespace LeftBound102298
def owner : Owner := ⟨.program ⟨257⟩, ⟨31091⟩⟩
def transferEvent : Nat := 102298
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102292 .summary, .result 102114 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 102292 .summary)
      LeftBound102126.bound (LeftBound102126.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨29935⟩⟩) (rawTerms := some (Proof.Events399.exact102292RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102126.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 102114 .summary)
      LeftBound102109.bound (LeftBound102109.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31090⟩⟩) (rawTerms := some (Proof.Events398.exact102114RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102109.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102126.bound, LeftBound102109.bound]
def bound : CoeffClass := .finite ⟨32192146870060392302605751287808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102126.bound, LeftBound102109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound102126.actual selector witness, LeftBound102109.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102298

namespace LeftBound102302
def owner : Owner := ⟨.program ⟨257⟩, ⟨31092⟩⟩
def transferEvent : Nat := 102302
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 102300 .coefficient) (.predecessor 1 102301 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102300 .coefficient)
      LeftBound102295.bound (LeftBound102295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102301 .coefficient)
      LeftBound15661.bound (LeftBound15661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15661.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound102295.bound LeftBound15661.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102295.bound, LeftBound15661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound102295.actual selector witness) * (LeftBound15661.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102302

namespace LeftBound102303
def owner : Owner := ⟨.program ⟨257⟩, ⟨31092⟩⟩
def transferEvent : Nat := 102303
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩ [⟨.result 15658 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15658 .coefficient)
      LeftAuthority15657.bound (LeftAuthority15657.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7167⟩⟩) (rawTerms := some (Proof.Events061.exact15658RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15657.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15657.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15657.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound102303

namespace LeftBound102304
def owner : Owner := ⟨.program ⟨257⟩, ⟨31092⟩⟩
def transferEvent : Nat := 102304
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 102299 .summary) (.transfer 102303) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 102299 .summary)
      LeftBound102298.bound (LeftBound102298.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31091⟩⟩) (rawTerms := some (Proof.Events399.exact102299RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102298.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 102303)
      LeftBound102303.bound (LeftBound102303.actual selector witness) := by
  exact .transfer (LeftBound102303.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound102298.bound LeftBound102303.bound
def bound : CoeffClass := .finite ⟨345660544987345366211554593406613108817920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102298.bound, LeftBound102303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound102298.actual selector witness) * (LeftBound102303.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102304

namespace LeftBound102319
def owner : Owner := ⟨.program ⟨257⟩, ⟨28410⟩⟩
def transferEvent : Nat := 102319
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 102317 .coefficient) (.predecessor 1 102318 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 102317 .coefficient)
      LeftBound94176.bound (LeftBound94176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 102318 .coefficient)
      LeftAuthority102315.bound (LeftAuthority102315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102315.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound94176.bound LeftAuthority102315.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94176.bound, LeftAuthority102315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound94176.actual selector witness) * (LeftAuthority102315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102319

namespace LeftBound102320
def owner : Owner := ⟨.program ⟨257⟩, ⟨28410⟩⟩
def transferEvent : Nat := 102320
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨28408⟩⟩]⟩ [⟨.result 102316 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 102316 .coefficient)
      LeftAuthority102315.bound (LeftAuthority102315.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨28408⟩⟩) (rawTerms := some (Proof.Events399.exact102316RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102315.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102315.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority102315.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority102315.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound102320

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
