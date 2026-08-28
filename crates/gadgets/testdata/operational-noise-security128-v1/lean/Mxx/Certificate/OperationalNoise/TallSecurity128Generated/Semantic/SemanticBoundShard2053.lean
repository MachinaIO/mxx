import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2052

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound301765
def owner : Owner := ⟨.program ⟨257⟩, ⟨21255⟩⟩
def transferEvent : Nat := 301765
def frameStart : Nat := 301748
def rule : BoundRule := .product (.predecessor 0 301763 .coefficient) (.predecessor 1 301764 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301763 .coefficient)
      LeftAuthority301761.bound (LeftAuthority301761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301761.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301764 .coefficient)
      LeftAuthority301758.bound (LeftAuthority301758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301758.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority301761.bound LeftAuthority301758.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301761.bound, LeftAuthority301758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority301761.actual selector witness) * (LeftAuthority301758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301765

namespace LeftBound301769
def owner : Owner := ⟨.program ⟨257⟩, ⟨21256⟩⟩
def transferEvent : Nat := 301769
def frameStart : Nat := 301748
def rule : BoundRule := .identity (.predecessor 0 301768 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301768 .coefficient)
      LeftBound301765.bound (LeftBound301765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301765.derived selector witness)

def rawBound : CoeffClass := LeftBound301765.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound301765.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound301769

namespace LeftBound301786
def owner : Owner := ⟨.program ⟨257⟩, ⟨23166⟩⟩
def transferEvent : Nat := 301786
def frameStart : Nat := 301748
def rule : BoundRule := .sum [.predecessor 0 301784 .coefficient, .predecessor 1 301785 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301784 .coefficient)
      LeftBound301769.bound (LeftBound301769.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound301769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301785 .coefficient)
      LeftAuthority301782.bound (LeftAuthority301782.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority301782.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301769.bound, LeftAuthority301782.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301769.bound, LeftAuthority301782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301769.actual selector witness, LeftAuthority301782.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301786

namespace LeftBound301789
def owner : Owner := ⟨.program ⟨257⟩, ⟨23167⟩⟩
def transferEvent : Nat := 301789
def frameStart : Nat := 301748
def rule : BoundRule := .identity (.predecessor 0 301788 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301788 .coefficient)
      LeftBound301786.bound (LeftBound301786.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound301786.derived selector witness)

def rawBound : CoeffClass := LeftBound301786.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound301786.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound301789

namespace LeftBound301795
def owner : Owner := ⟨.program ⟨257⟩, ⟨23168⟩⟩
def transferEvent : Nat := 301795
def frameStart : Nat := 301748
def rule : BoundRule := .product (.predecessor 0 301793 .coefficient) (.predecessor 1 301794 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301793 .coefficient)
      LeftAuthority301791.bound (LeftAuthority301791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301794 .coefficient)
      LeftBound301789.bound (LeftBound301789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301789.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority301791.bound LeftBound301789.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301791.bound, LeftBound301789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority301791.actual selector witness) * (LeftBound301789.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301795

namespace LeftBound301811
def owner : Owner := ⟨.program ⟨257⟩, ⟨9575⟩⟩
def transferEvent : Nat := 301811
def frameStart : Nat := 301748
def rule : BoundRule := .scale (.predecessor 0 301809 .coefficient) (.value (.predecessor 1 301810 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301809 .coefficient)
      LeftAuthority301807.bound (LeftAuthority301807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301810 .coefficient)
      LeftAuthority301798.bound (LeftAuthority301798.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority301798.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority301807.bound LeftAuthority301798.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301807.bound, LeftAuthority301798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority301807.actual selector witness) * (LeftAuthority301798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound301811

namespace LeftBound301814
def owner : Owner := ⟨.program ⟨257⟩, ⟨7286⟩⟩
def transferEvent : Nat := 301814
def frameStart : Nat := 301748
def rule : BoundRule := .identity (.predecessor 0 301813 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301813 .coefficient)
      LeftAuthority301801.bound (LeftAuthority301801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301801.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301801.derived selector witness)

def rawBound : CoeffClass := LeftAuthority301801.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority301801.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound301814

namespace LeftBound301818
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def transferEvent : Nat := 301818
def frameStart : Nat := 301748
def rule : BoundRule := .product (.predecessor 0 301816 .coefficient) (.predecessor 1 301817 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301816 .coefficient)
      LeftBound301814.bound (LeftBound301814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301817 .coefficient)
      LeftBound301811.bound (LeftBound301811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301811.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301811.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound301814.bound LeftBound301811.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301814.bound, LeftBound301811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound301814.actual selector witness) * (LeftBound301811.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301818

namespace LeftBound301823
def owner : Owner := ⟨.program ⟨257⟩, ⟨23169⟩⟩
def transferEvent : Nat := 301823
def frameStart : Nat := 301748
def rule : BoundRule := .sum [.predecessor 0 301821 .coefficient, .predecessor 1 301822 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301821 .coefficient)
      LeftBound301818.bound (LeftBound301818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301818.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301822 .coefficient)
      LeftBound301795.bound (LeftBound301795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301795.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301818.bound, LeftBound301795.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301818.bound, LeftBound301795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301818.actual selector witness, LeftBound301795.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301823

namespace LeftBound301827
def owner : Owner := ⟨.program ⟨257⟩, ⟨23332⟩⟩
def transferEvent : Nat := 301827
def frameStart : Nat := 301748
def rule : BoundRule := .product (.predecessor 0 301825 .coefficient) (.predecessor 1 301826 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301825 .coefficient)
      LeftBound301823.bound (LeftBound301823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact301824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301826 .coefficient)
      LeftAuthority301780.bound (LeftAuthority301780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301780.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301780.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound301823.bound LeftAuthority301780.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301823.bound, LeftAuthority301780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound301823.actual selector witness) * (LeftAuthority301780.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301827

namespace LeftBound301838
def owner : Owner := ⟨.program ⟨257⟩, ⟨21730⟩⟩
def transferEvent : Nat := 301838
def frameStart : Nat := 301748
def rule : BoundRule := .product (.predecessor 0 301836 .coefficient) (.predecessor 1 301837 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301836 .coefficient)
      LeftAuthority301791.bound (LeftAuthority301791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301837 .coefficient)
      LeftAuthority301834.bound (LeftAuthority301834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact301835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301834.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority301791.bound LeftAuthority301834.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301791.bound, LeftAuthority301834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority301791.actual selector witness) * (LeftAuthority301834.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301838

namespace LeftBound301846
def owner : Owner := ⟨.program ⟨257⟩, ⟨21731⟩⟩
def transferEvent : Nat := 301846
def frameStart : Nat := 301748
def rule : BoundRule := .sum [.predecessor 0 301844 .coefficient, .predecessor 1 301845 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301844 .coefficient)
      LeftAuthority301842.bound (LeftAuthority301842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact301843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301845 .coefficient)
      LeftBound301838.bound (LeftBound301838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact301840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority301842.bound, LeftBound301838.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301842.bound, LeftBound301838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority301842.actual selector witness, LeftBound301838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301846

namespace LeftBound301850
def owner : Owner := ⟨.program ⟨257⟩, ⟨23333⟩⟩
def transferEvent : Nat := 301850
def frameStart : Nat := 301748
def rule : BoundRule := .sum [.predecessor 0 301848 .coefficient, .predecessor 1 301849 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301848 .coefficient)
      LeftBound301846.bound (LeftBound301846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact301847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301849 .coefficient)
      LeftBound301827.bound (LeftBound301827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact301832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301827.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301827.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301846.bound, LeftBound301827.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301846.bound, LeftBound301827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301846.actual selector witness, LeftBound301827.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301850

namespace LeftBound301863
def owner : Owner := ⟨.program ⟨257⟩, ⟨23331⟩⟩
def transferEvent : Nat := 301863
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 301861 .coefficient, .predecessor 1 301862 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301861 .coefficient)
      LeftBound301708.bound (LeftBound301708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact301860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301862 .coefficient)
      LeftBound301691.bound (LeftBound301691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301691.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301708.bound, LeftBound301691.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301708.bound, LeftBound301691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301708.actual selector witness, LeftBound301691.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301863

namespace LeftBound301866
def owner : Owner := ⟨.program ⟨257⟩, ⟨23331⟩⟩
def transferEvent : Nat := 301866
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 301860 .summary, .result 301698 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301860 .summary)
      LeftBound301710.bound (LeftBound301710.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22272⟩⟩) (rawTerms := some (Proof.Events1179.exact301860RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301710.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301698 .summary)
      LeftBound301693.bound (LeftBound301693.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23330⟩⟩) (rawTerms := some (Proof.Events1178.exact301698RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301710.bound, LeftBound301693.bound]
def bound : CoeffClass := .finite ⟨2997834576566628384768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301710.bound, LeftBound301693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301710.actual selector witness, LeftBound301693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301866

namespace LeftBound301870
def owner : Owner := ⟨.program ⟨257⟩, ⟨23564⟩⟩
def transferEvent : Nat := 301870
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 301868 .coefficient) (.predecessor 1 301869 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301868 .coefficient)
      LeftBound301863.bound (LeftBound301863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact301867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301869 .coefficient)
      LeftAuthority301613.bound (LeftAuthority301613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1178.exact301614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301613.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301613.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound301863.bound LeftAuthority301613.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301863.bound, LeftAuthority301613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound301863.actual selector witness) * (LeftAuthority301613.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301870

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
