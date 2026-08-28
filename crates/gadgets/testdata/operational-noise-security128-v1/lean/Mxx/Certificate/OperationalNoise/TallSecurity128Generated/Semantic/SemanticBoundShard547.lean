import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard546

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound85821
def owner : Owner := ⟨.program ⟨257⟩, ⟨67026⟩⟩
def transferEvent : Nat := 85821
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85819 .coefficient, .predecessor 1 85820 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85819 .coefficient)
      LeftBound85817.bound (LeftBound85817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85820 .coefficient)
      LeftAuthority85470.bound (LeftAuthority85470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85470.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85817.bound, LeftAuthority85470.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85817.bound, LeftAuthority85470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85817.actual selector witness, LeftAuthority85470.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85821

namespace LeftBound85825
def owner : Owner := ⟨.program ⟨257⟩, ⟨67027⟩⟩
def transferEvent : Nat := 85825
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85823 .coefficient, .predecessor 1 85824 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85823 .coefficient)
      LeftBound85821.bound (LeftBound85821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85824 .coefficient)
      LeftAuthority85447.bound (LeftAuthority85447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85447.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85821.bound, LeftAuthority85447.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85821.bound, LeftAuthority85447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85821.actual selector witness, LeftAuthority85447.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85825

namespace LeftBound85829
def owner : Owner := ⟨.program ⟨257⟩, ⟨67028⟩⟩
def transferEvent : Nat := 85829
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85827 .coefficient, .predecessor 1 85828 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85827 .coefficient)
      LeftBound85825.bound (LeftBound85825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85828 .coefficient)
      LeftAuthority85424.bound (LeftAuthority85424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85424.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85825.bound, LeftAuthority85424.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85825.bound, LeftAuthority85424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85825.actual selector witness, LeftAuthority85424.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85829

namespace LeftBound85833
def owner : Owner := ⟨.program ⟨257⟩, ⟨67029⟩⟩
def transferEvent : Nat := 85833
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85831 .coefficient, .predecessor 1 85832 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85831 .coefficient)
      LeftBound85829.bound (LeftBound85829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85829.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85832 .coefficient)
      LeftAuthority85401.bound (LeftAuthority85401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85829.bound, LeftAuthority85401.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85829.bound, LeftAuthority85401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85829.actual selector witness, LeftAuthority85401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85833

namespace LeftBound85837
def owner : Owner := ⟨.program ⟨257⟩, ⟨67030⟩⟩
def transferEvent : Nat := 85837
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85835 .coefficient, .predecessor 1 85836 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85835 .coefficient)
      LeftBound85833.bound (LeftBound85833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85836 .coefficient)
      LeftAuthority85378.bound (LeftAuthority85378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85833.bound, LeftAuthority85378.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85833.bound, LeftAuthority85378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85833.actual selector witness, LeftAuthority85378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85837

namespace LeftBound85840
def owner : Owner := ⟨.program ⟨257⟩, ⟨67031⟩⟩
def transferEvent : Nat := 85840
def frameStart : Nat := 85336
def rule : BoundRule := .identity (.predecessor 0 85839 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85839 .coefficient)
      LeftBound85837.bound (LeftBound85837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85837.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85837.derived selector witness)

def rawBound : CoeffClass := LeftBound85837.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound85837.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85840

namespace LeftBound85857
def owner : Owner := ⟨.program ⟨257⟩, ⟨69111⟩⟩
def transferEvent : Nat := 85857
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85855 .coefficient, .predecessor 1 85856 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85855 .coefficient)
      LeftBound85840.bound (LeftBound85840.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound85840.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85856 .coefficient)
      LeftAuthority85853.bound (LeftAuthority85853.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority85853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85840.bound, LeftAuthority85853.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85840.bound, LeftAuthority85853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85840.actual selector witness, LeftAuthority85853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85857

namespace LeftBound85860
def owner : Owner := ⟨.program ⟨257⟩, ⟨69112⟩⟩
def transferEvent : Nat := 85860
def frameStart : Nat := 85336
def rule : BoundRule := .identity (.predecessor 0 85859 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85859 .coefficient)
      LeftBound85857.bound (LeftBound85857.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound85857.derived selector witness)

def rawBound : CoeffClass := LeftBound85857.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound85857.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85860

namespace LeftBound85866
def owner : Owner := ⟨.program ⟨257⟩, ⟨69113⟩⟩
def transferEvent : Nat := 85866
def frameStart : Nat := 85336
def rule : BoundRule := .product (.predecessor 0 85864 .coefficient) (.predecessor 1 85865 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85864 .coefficient)
      LeftAuthority85862.bound (LeftAuthority85862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85865 .coefficient)
      LeftBound85860.bound (LeftBound85860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority85862.bound LeftBound85860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85862.bound, LeftBound85860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority85862.actual selector witness) * (LeftBound85860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85866

namespace LeftBound85942
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 85942
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85940 .coefficient, .predecessor 1 85941 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85940 .coefficient)
      LeftAuthority85938.bound (LeftAuthority85938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85938.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85941 .coefficient)
      LeftAuthority85935.bound (LeftAuthority85935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85935.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority85938.bound, LeftAuthority85935.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85938.bound, LeftAuthority85935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority85938.actual selector witness, LeftAuthority85935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85942

namespace LeftBound85946
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 85946
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85944 .coefficient, .predecessor 1 85945 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85944 .coefficient)
      LeftBound85942.bound (LeftBound85942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85945 .coefficient)
      LeftAuthority85932.bound (LeftAuthority85932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85932.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85942.bound, LeftAuthority85932.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85942.bound, LeftAuthority85932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85942.actual selector witness, LeftAuthority85932.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85946

namespace LeftBound85950
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 85950
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85948 .coefficient, .predecessor 1 85949 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85948 .coefficient)
      LeftBound85946.bound (LeftBound85946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85949 .coefficient)
      LeftAuthority85929.bound (LeftAuthority85929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85929.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85946.bound, LeftAuthority85929.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85946.bound, LeftAuthority85929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85946.actual selector witness, LeftAuthority85929.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85950

namespace LeftBound85954
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 85954
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85952 .coefficient, .predecessor 1 85953 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85952 .coefficient)
      LeftBound85950.bound (LeftBound85950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85950.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85953 .coefficient)
      LeftAuthority85926.bound (LeftAuthority85926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85926.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85950.bound, LeftAuthority85926.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85950.bound, LeftAuthority85926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85950.actual selector witness, LeftAuthority85926.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85954

namespace LeftBound85958
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 85958
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85956 .coefficient, .predecessor 1 85957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85956 .coefficient)
      LeftBound85954.bound (LeftBound85954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85957 .coefficient)
      LeftAuthority85923.bound (LeftAuthority85923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85923.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85954.bound, LeftAuthority85923.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85954.bound, LeftAuthority85923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85954.actual selector witness, LeftAuthority85923.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85958

namespace LeftBound85962
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 85962
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85960 .coefficient, .predecessor 1 85961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85960 .coefficient)
      LeftBound85958.bound (LeftBound85958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85961 .coefficient)
      LeftAuthority85920.bound (LeftAuthority85920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85920.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85958.bound, LeftAuthority85920.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85958.bound, LeftAuthority85920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85958.actual selector witness, LeftAuthority85920.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85962

namespace LeftBound85966
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 85966
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85964 .coefficient, .predecessor 1 85965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85964 .coefficient)
      LeftBound85962.bound (LeftBound85962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85965 .coefficient)
      LeftAuthority85917.bound (LeftAuthority85917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85962.bound, LeftAuthority85917.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85962.bound, LeftAuthority85917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85962.actual selector witness, LeftAuthority85917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85966

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
