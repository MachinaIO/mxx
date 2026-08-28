import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1595

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound236978
def owner : Owner := ⟨.program ⟨257⟩, ⟨49419⟩⟩
def transferEvent : Nat := 236978
def frameStart : Nat := 236925
def rule : BoundRule := .identity (.predecessor 0 236977 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236977 .coefficient)
      LeftBound236975.bound (LeftBound236975.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound236975.derived selector witness)

def rawBound : CoeffClass := LeftBound236975.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236975.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound236975.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound236978

namespace LeftBound236984
def owner : Owner := ⟨.program ⟨257⟩, ⟨49420⟩⟩
def transferEvent : Nat := 236984
def frameStart : Nat := 236925
def rule : BoundRule := .product (.predecessor 0 236982 .coefficient) (.predecessor 1 236983 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236982 .coefficient)
      LeftAuthority236980.bound (LeftAuthority236980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236980.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236983 .coefficient)
      LeftBound236978.bound (LeftBound236978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority236980.bound LeftBound236978.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority236980.bound, LeftBound236978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority236980.actual selector witness) * (LeftBound236978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound236984

namespace LeftBound237000
def owner : Owner := ⟨.program ⟨257⟩, ⟨9566⟩⟩
def transferEvent : Nat := 237000
def frameStart : Nat := 236925
def rule : BoundRule := .scale (.predecessor 0 236998 .coefficient) (.value (.predecessor 1 236999 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236998 .coefficient)
      LeftAuthority236996.bound (LeftAuthority236996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236999 .coefficient)
      LeftAuthority236987.bound (LeftAuthority236987.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority236987.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority236996.bound LeftAuthority236987.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority236996.bound, LeftAuthority236987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority236996.actual selector witness) * (LeftAuthority236987.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound237000

namespace LeftBound237003
def owner : Owner := ⟨.program ⟨257⟩, ⟨7302⟩⟩
def transferEvent : Nat := 237003
def frameStart : Nat := 236925
def rule : BoundRule := .identity (.predecessor 0 237002 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237002 .coefficient)
      LeftAuthority236990.bound (LeftAuthority236990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236990.derived selector witness)

def rawBound : CoeffClass := LeftAuthority236990.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority236990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority236990.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound237003

namespace LeftBound237007
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def transferEvent : Nat := 237007
def frameStart : Nat := 236925
def rule : BoundRule := .product (.predecessor 0 237005 .coefficient) (.predecessor 1 237006 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237005 .coefficient)
      LeftBound237003.bound (LeftBound237003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237006 .coefficient)
      LeftBound237000.bound (LeftBound237000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound237003.bound LeftBound237000.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237003.bound, LeftBound237000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound237003.actual selector witness) * (LeftBound237000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237007

namespace LeftBound237012
def owner : Owner := ⟨.program ⟨257⟩, ⟨49421⟩⟩
def transferEvent : Nat := 237012
def frameStart : Nat := 236925
def rule : BoundRule := .sum [.predecessor 0 237010 .coefficient, .predecessor 1 237011 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237010 .coefficient)
      LeftBound237007.bound (LeftBound237007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237007.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237011 .coefficient)
      LeftBound236984.bound (LeftBound236984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236984.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound237007.bound, LeftBound236984.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237007.bound, LeftBound236984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound237007.actual selector witness, LeftBound236984.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237012

namespace LeftBound237016
def owner : Owner := ⟨.program ⟨257⟩, ⟨49640⟩⟩
def transferEvent : Nat := 237016
def frameStart : Nat := 236925
def rule : BoundRule := .product (.predecessor 0 237014 .coefficient) (.predecessor 1 237015 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237014 .coefficient)
      LeftBound237012.bound (LeftBound237012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237015 .coefficient)
      LeftAuthority236969.bound (LeftAuthority236969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236969.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236969.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound237012.bound LeftAuthority236969.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237012.bound, LeftAuthority236969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound237012.actual selector witness) * (LeftAuthority236969.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237016

namespace LeftBound237027
def owner : Owner := ⟨.program ⟨257⟩, ⟨48134⟩⟩
def transferEvent : Nat := 237027
def frameStart : Nat := 236925
def rule : BoundRule := .product (.predecessor 0 237025 .coefficient) (.predecessor 1 237026 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237025 .coefficient)
      LeftAuthority236980.bound (LeftAuthority236980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236980.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237026 .coefficient)
      LeftAuthority237023.bound (LeftAuthority237023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237023.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority236980.bound LeftAuthority237023.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority236980.bound, LeftAuthority237023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority236980.actual selector witness) * (LeftAuthority237023.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237027

namespace LeftBound237035
def owner : Owner := ⟨.program ⟨257⟩, ⟨48135⟩⟩
def transferEvent : Nat := 237035
def frameStart : Nat := 236925
def rule : BoundRule := .sum [.predecessor 0 237033 .coefficient, .predecessor 1 237034 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237033 .coefficient)
      LeftAuthority237031.bound (LeftAuthority237031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237034 .coefficient)
      LeftBound237027.bound (LeftBound237027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237027.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority237031.bound, LeftBound237027.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority237031.bound, LeftBound237027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority237031.actual selector witness, LeftBound237027.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237035

namespace LeftBound237039
def owner : Owner := ⟨.program ⟨257⟩, ⟨49641⟩⟩
def transferEvent : Nat := 237039
def frameStart : Nat := 236925
def rule : BoundRule := .sum [.predecessor 0 237037 .coefficient, .predecessor 1 237038 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237037 .coefficient)
      LeftBound237035.bound (LeftBound237035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237038 .coefficient)
      LeftBound237016.bound (LeftBound237016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound237035.bound, LeftBound237016.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237035.bound, LeftBound237016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound237035.actual selector witness, LeftBound237016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237039

namespace LeftBound237052
def owner : Owner := ⟨.program ⟨257⟩, ⟨49639⟩⟩
def transferEvent : Nat := 237052
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 237050 .coefficient, .predecessor 1 237051 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237050 .coefficient)
      LeftBound236873.bound (LeftBound236873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact237049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237051 .coefficient)
      LeftBound236845.bound (LeftBound236845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236873.bound, LeftBound236845.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236873.bound, LeftBound236845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236873.actual selector witness, LeftBound236845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237052

namespace LeftBound237055
def owner : Owner := ⟨.program ⟨257⟩, ⟨49639⟩⟩
def transferEvent : Nat := 237055
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 237049 .summary, .result 236852 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 237049 .summary)
      LeftBound236875.bound (LeftBound236875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48572⟩⟩) (rawTerms := some (Proof.Events925.exact237049RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236852 .summary)
      LeftBound236847.bound (LeftBound236847.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49638⟩⟩) (rawTerms := some (Proof.Events925.exact236852RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236847.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236875.bound, LeftBound236847.bound]
def bound : CoeffClass := .finite ⟨2998346861024241778688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236875.bound, LeftBound236847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236875.actual selector witness, LeftBound236847.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237055

namespace LeftBound237059
def owner : Owner := ⟨.program ⟨257⟩, ⟨49981⟩⟩
def transferEvent : Nat := 237059
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 237057 .coefficient) (.predecessor 1 237058 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237057 .coefficient)
      LeftBound237052.bound (LeftBound237052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events926.exact237056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237058 .coefficient)
      LeftAuthority236762.bound (LeftAuthority236762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236762.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236762.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound237052.bound LeftAuthority236762.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237052.bound, LeftAuthority236762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound237052.actual selector witness) * (LeftAuthority236762.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237059

namespace LeftBound237060
def owner : Owner := ⟨.program ⟨257⟩, ⟨49981⟩⟩
def transferEvent : Nat := 237060
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩ [⟨.result 236763 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236763 .coefficient)
      LeftAuthority236762.bound (LeftAuthority236762.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨49979⟩⟩) (rawTerms := some (Proof.Events924.exact236763RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236762.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236762.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority236762.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority236762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority236762.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound237060

namespace LeftBound237061
def owner : Owner := ⟨.program ⟨257⟩, ⟨49981⟩⟩
def transferEvent : Nat := 237061
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 237056 .summary) (.transfer 237060) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 237056 .summary)
      LeftBound237055.bound (LeftBound237055.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49639⟩⟩) (rawTerms := some (Proof.Events926.exact237056RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound237055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 237060)
      LeftBound237060.bound (LeftBound237060.actual selector witness) := by
  exact .transfer (LeftBound237060.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound237055.bound LeftBound237060.bound
def bound : CoeffClass := .finite ⟨32194504275408438756654574469120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237055.bound, LeftBound237060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound237055.actual selector witness) * (LeftBound237060.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237061

namespace LeftBound237072
def owner : Owner := ⟨.program ⟨257⟩, ⟨48858⟩⟩
def transferEvent : Nat := 237072
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 237070 .coefficient) (.value (.predecessor 1 237071 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237070 .coefficient)
      LeftAuthority237068.bound (LeftAuthority237068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events926.exact237069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237068.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237071 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority237068.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority237068.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority237068.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound237072

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
