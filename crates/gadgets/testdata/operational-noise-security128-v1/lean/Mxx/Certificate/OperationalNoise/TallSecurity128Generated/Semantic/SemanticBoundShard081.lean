import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard080

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound18802
def owner : Owner := ⟨.program ⟨257⟩, ⟨9557⟩⟩
def transferEvent : Nat := 18802
def frameStart : Nat := 18727
def rule : BoundRule := .scale (.predecessor 0 18800 .coefficient) (.value (.predecessor 1 18801 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18800 .coefficient)
      LeftAuthority18798.bound (LeftAuthority18798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18801 .coefficient)
      LeftAuthority18789.bound (LeftAuthority18789.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority18789.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority18798.bound LeftAuthority18789.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18798.bound, LeftAuthority18789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority18798.actual selector witness) * (LeftAuthority18789.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound18802

namespace LeftBound18805
def owner : Owner := ⟨.program ⟨257⟩, ⟨7299⟩⟩
def transferEvent : Nat := 18805
def frameStart : Nat := 18727
def rule : BoundRule := .identity (.predecessor 0 18804 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18804 .coefficient)
      LeftAuthority18792.bound (LeftAuthority18792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18792.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18792.derived selector witness)

def rawBound : CoeffClass := LeftAuthority18792.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18792.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority18792.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18805

namespace LeftBound18809
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def transferEvent : Nat := 18809
def frameStart : Nat := 18727
def rule : BoundRule := .product (.predecessor 0 18807 .coefficient) (.predecessor 1 18808 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18807 .coefficient)
      LeftBound18805.bound (LeftBound18805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18805.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18808 .coefficient)
      LeftBound18802.bound (LeftBound18802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18802.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound18805.bound LeftBound18802.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18805.bound, LeftBound18802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound18805.actual selector witness) * (LeftBound18802.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18809

namespace LeftBound18814
def owner : Owner := ⟨.program ⟨257⟩, ⟨41353⟩⟩
def transferEvent : Nat := 18814
def frameStart : Nat := 18727
def rule : BoundRule := .sum [.predecessor 0 18812 .coefficient, .predecessor 1 18813 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18812 .coefficient)
      LeftBound18809.bound (LeftBound18809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18813 .coefficient)
      LeftBound18786.bound (LeftBound18786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18786.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18786.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18809.bound, LeftBound18786.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18809.bound, LeftBound18786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound18809.actual selector witness, LeftBound18786.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18814

namespace LeftBound18818
def owner : Owner := ⟨.program ⟨257⟩, ⟨41526⟩⟩
def transferEvent : Nat := 18818
def frameStart : Nat := 18727
def rule : BoundRule := .product (.predecessor 0 18816 .coefficient) (.predecessor 1 18817 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18816 .coefficient)
      LeftBound18814.bound (LeftBound18814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18817 .coefficient)
      LeftAuthority18771.bound (LeftAuthority18771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18771.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18771.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound18814.bound LeftAuthority18771.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18814.bound, LeftAuthority18771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound18814.actual selector witness) * (LeftAuthority18771.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18818

namespace LeftBound18829
def owner : Owner := ⟨.program ⟨257⟩, ⟨40040⟩⟩
def transferEvent : Nat := 18829
def frameStart : Nat := 18727
def rule : BoundRule := .product (.predecessor 0 18827 .coefficient) (.predecessor 1 18828 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18827 .coefficient)
      LeftAuthority18782.bound (LeftAuthority18782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18782.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18828 .coefficient)
      LeftAuthority18825.bound (LeftAuthority18825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18825.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18825.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority18782.bound LeftAuthority18825.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18782.bound, LeftAuthority18825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority18782.actual selector witness) * (LeftAuthority18825.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18829

namespace LeftBound18837
def owner : Owner := ⟨.program ⟨257⟩, ⟨40041⟩⟩
def transferEvent : Nat := 18837
def frameStart : Nat := 18727
def rule : BoundRule := .sum [.predecessor 0 18835 .coefficient, .predecessor 1 18836 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18835 .coefficient)
      LeftAuthority18833.bound (LeftAuthority18833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18833.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18836 .coefficient)
      LeftBound18829.bound (LeftBound18829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18829.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18829.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18833.bound, LeftBound18829.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18833.bound, LeftBound18829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority18833.actual selector witness, LeftBound18829.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18837

namespace LeftBound18841
def owner : Owner := ⟨.program ⟨257⟩, ⟨41527⟩⟩
def transferEvent : Nat := 18841
def frameStart : Nat := 18727
def rule : BoundRule := .sum [.predecessor 0 18839 .coefficient, .predecessor 1 18840 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18839 .coefficient)
      LeftBound18837.bound (LeftBound18837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18837.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18840 .coefficient)
      LeftBound18818.bound (LeftBound18818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18818.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18837.bound, LeftBound18818.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18837.bound, LeftBound18818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound18837.actual selector witness, LeftBound18818.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18841

namespace LeftBound18854
def owner : Owner := ⟨.program ⟨257⟩, ⟨41525⟩⟩
def transferEvent : Nat := 18854
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 18852 .coefficient, .predecessor 1 18853 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18852 .coefficient)
      LeftBound18675.bound (LeftBound18675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18853 .coefficient)
      LeftBound18658.bound (LeftBound18658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18675.bound, LeftBound18658.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18675.bound, LeftBound18658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound18675.actual selector witness, LeftBound18658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18854

namespace LeftBound18857
def owner : Owner := ⟨.program ⟨257⟩, ⟨41525⟩⟩
def transferEvent : Nat := 18857
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 18851 .summary, .result 18665 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18851 .summary)
      LeftBound18677.bound (LeftBound18677.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40465⟩⟩) (rawTerms := some (Proof.Events073.exact18851RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18665 .summary)
      LeftBound18660.bound (LeftBound18660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41524⟩⟩) (rawTerms := some (Proof.Events072.exact18665RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18660.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18677.bound, LeftBound18660.bound]
def bound : CoeffClass := .finite ⟨2998218789909838430208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18677.bound, LeftBound18660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound18677.actual selector witness, LeftBound18660.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18857

namespace LeftBound18861
def owner : Owner := ⟨.program ⟨257⟩, ⟨41773⟩⟩
def transferEvent : Nat := 18861
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18859 .coefficient) (.predecessor 1 18860 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18859 .coefficient)
      LeftBound18854.bound (LeftBound18854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18854.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18860 .coefficient)
      LeftAuthority18561.bound (LeftAuthority18561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18561.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18561.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound18854.bound LeftAuthority18561.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18854.bound, LeftAuthority18561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound18854.actual selector witness) * (LeftAuthority18561.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18861

namespace LeftBound18862
def owner : Owner := ⟨.program ⟨257⟩, ⟨41773⟩⟩
def transferEvent : Nat := 18862
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨41771⟩⟩]⟩ [⟨.result 18562 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18562 .coefficient)
      LeftAuthority18561.bound (LeftAuthority18561.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨41771⟩⟩) (rawTerms := some (Proof.Events072.exact18562RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18561.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18561.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority18561.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority18561.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18862

namespace LeftBound18863
def owner : Owner := ⟨.program ⟨257⟩, ⟨41773⟩⟩
def transferEvent : Nat := 18863
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 18858 .summary) (.transfer 18862) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18858 .summary)
      LeftBound18857.bound (LeftBound18857.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41525⟩⟩) (rawTerms := some (Proof.Events073.exact18858RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 18862)
      LeftBound18862.bound (LeftBound18862.actual selector witness) := by
  exact .transfer (LeftBound18862.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound18857.bound LeftBound18862.bound
def bound : CoeffClass := .finite ⟨32193129122288627115968346193920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18857.bound, LeftBound18862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound18857.actual selector witness) * (LeftBound18862.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18863

namespace LeftBound18874
def owner : Owner := ⟨.program ⟨257⟩, ⟨40684⟩⟩
def transferEvent : Nat := 18874
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 18872 .coefficient) (.value (.predecessor 1 18873 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18872 .coefficient)
      LeftAuthority18870.bound (LeftAuthority18870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18873 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority18870.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18870.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority18870.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound18874

namespace LeftBound18878
def owner : Owner := ⟨.program ⟨257⟩, ⟨40685⟩⟩
def transferEvent : Nat := 18878
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18876 .coefficient) (.predecessor 1 18877 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 18876 .coefficient)
      LeftBound17166.bound (LeftBound17166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 18877 .coefficient)
      LeftBound18874.bound (LeftBound18874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18874.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17166.bound LeftBound18874.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17166.bound, LeftBound18874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17166.actual selector witness) * (LeftBound18874.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18878

namespace LeftBound18879
def owner : Owner := ⟨.program ⟨257⟩, ⟨40685⟩⟩
def transferEvent : Nat := 18879
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨40682⟩⟩]⟩ [⟨.result 18871 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18871 .coefficient)
      LeftAuthority18870.bound (LeftAuthority18870.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨40682⟩⟩) (rawTerms := some (Proof.Events073.exact18871RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18870.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority18870.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority18870.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18879

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
