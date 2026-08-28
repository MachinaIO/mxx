import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1088
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1117

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound167734
def owner : Owner := ⟨.program ⟨257⟩, ⟨7294⟩⟩
def transferEvent : Nat := 167734
def frameStart : Nat := 167656
def rule : BoundRule := .identity (.predecessor 0 167733 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167733 .coefficient)
      LeftAuthority167721.bound (LeftAuthority167721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167721.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167721.derived selector witness)

def rawBound : CoeffClass := LeftAuthority167721.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority167721.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound167734

namespace LeftBound167738
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def transferEvent : Nat := 167738
def frameStart : Nat := 167656
def rule : BoundRule := .product (.predecessor 0 167736 .coefficient) (.predecessor 1 167737 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167736 .coefficient)
      LeftBound167734.bound (LeftBound167734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167734.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167737 .coefficient)
      LeftBound167731.bound (LeftBound167731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167731.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound167734.bound LeftBound167731.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167734.bound, LeftBound167731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound167734.actual selector witness) * (LeftBound167731.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167738

namespace LeftBound167743
def owner : Owner := ⟨.program ⟨257⟩, ⟨68946⟩⟩
def transferEvent : Nat := 167743
def frameStart : Nat := 167656
def rule : BoundRule := .sum [.predecessor 0 167741 .coefficient, .predecessor 1 167742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167741 .coefficient)
      LeftBound167738.bound (LeftBound167738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167742 .coefficient)
      LeftBound167715.bound (LeftBound167715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167715.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound167738.bound, LeftBound167715.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167738.bound, LeftBound167715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound167738.actual selector witness, LeftBound167715.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167743

namespace LeftBound167747
def owner : Owner := ⟨.program ⟨257⟩, ⟨69287⟩⟩
def transferEvent : Nat := 167747
def frameStart : Nat := 167656
def rule : BoundRule := .product (.predecessor 0 167745 .coefficient) (.predecessor 1 167746 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167745 .coefficient)
      LeftBound167743.bound (LeftBound167743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167746 .coefficient)
      LeftAuthority167700.bound (LeftAuthority167700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167700.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167700.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound167743.bound LeftAuthority167700.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167743.bound, LeftAuthority167700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound167743.actual selector witness) * (LeftAuthority167700.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167747

namespace LeftBound167758
def owner : Owner := ⟨.program ⟨257⟩, ⟨65822⟩⟩
def transferEvent : Nat := 167758
def frameStart : Nat := 167656
def rule : BoundRule := .product (.predecessor 0 167756 .coefficient) (.predecessor 1 167757 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167756 .coefficient)
      LeftAuthority167711.bound (LeftAuthority167711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167711.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167757 .coefficient)
      LeftAuthority167754.bound (LeftAuthority167754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167754.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority167711.bound LeftAuthority167754.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167711.bound, LeftAuthority167754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority167711.actual selector witness) * (LeftAuthority167754.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167758

namespace LeftBound167766
def owner : Owner := ⟨.program ⟨257⟩, ⟨65823⟩⟩
def transferEvent : Nat := 167766
def frameStart : Nat := 167656
def rule : BoundRule := .sum [.predecessor 0 167764 .coefficient, .predecessor 1 167765 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167764 .coefficient)
      LeftAuthority167762.bound (LeftAuthority167762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167762.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167765 .coefficient)
      LeftBound167758.bound (LeftBound167758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167758.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority167762.bound, LeftBound167758.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167762.bound, LeftBound167758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority167762.actual selector witness, LeftBound167758.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167766

namespace LeftBound167770
def owner : Owner := ⟨.program ⟨257⟩, ⟨69288⟩⟩
def transferEvent : Nat := 167770
def frameStart : Nat := 167656
def rule : BoundRule := .sum [.predecessor 0 167768 .coefficient, .predecessor 1 167769 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167768 .coefficient)
      LeftBound167766.bound (LeftBound167766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167766.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167769 .coefficient)
      LeftBound167747.bound (LeftBound167747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound167766.bound, LeftBound167747.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167766.bound, LeftBound167747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound167766.actual selector witness, LeftBound167747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167770

namespace LeftBound167783
def owner : Owner := ⟨.program ⟨257⟩, ⟨69286⟩⟩
def transferEvent : Nat := 167783
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 167781 .coefficient, .predecessor 1 167782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167781 .coefficient)
      LeftBound167604.bound (LeftBound167604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167782 .coefficient)
      LeftBound167587.bound (LeftBound167587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events654.exact167594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167587.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound167604.bound, LeftBound167587.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167604.bound, LeftBound167587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound167604.actual selector witness, LeftBound167587.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167783

namespace LeftBound167786
def owner : Owner := ⟨.program ⟨257⟩, ⟨69286⟩⟩
def transferEvent : Nat := 167786
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 167780 .summary, .result 167594 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167780 .summary)
      LeftBound167606.bound (LeftBound167606.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67813⟩⟩) (rawTerms := some (Proof.Events655.exact167780RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound167606.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167594 .summary)
      LeftBound167589.bound (LeftBound167589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69285⟩⟩) (rawTerms := some (Proof.Events654.exact167594RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound167589.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound167606.bound, LeftBound167589.bound]
def bound : CoeffClass := .finite ⟨2998054127048462696448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167606.bound, LeftBound167589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound167606.actual selector witness, LeftBound167589.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound167786

namespace LeftBound167790
def owner : Owner := ⟨.program ⟨257⟩, ⟨70495⟩⟩
def transferEvent : Nat := 167790
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 167788 .coefficient) (.predecessor 1 167789 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167788 .coefficient)
      LeftBound167783.bound (LeftBound167783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167789 .coefficient)
      LeftAuthority167509.bound (LeftAuthority167509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events654.exact167510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167509.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167509.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound167783.bound LeftAuthority167509.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167783.bound, LeftAuthority167509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound167783.actual selector witness) * (LeftAuthority167509.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167790

namespace LeftBound167791
def owner : Owner := ⟨.program ⟨257⟩, ⟨70495⟩⟩
def transferEvent : Nat := 167791
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨70493⟩⟩]⟩ [⟨.result 167510 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167510 .coefficient)
      LeftAuthority167509.bound (LeftAuthority167509.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨70493⟩⟩) (rawTerms := some (Proof.Events654.exact167510RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167509.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167509.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority167509.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority167509.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound167791

namespace LeftBound167792
def owner : Owner := ⟨.program ⟨257⟩, ⟨70495⟩⟩
def transferEvent : Nat := 167792
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 167787 .summary) (.transfer 167791) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167787 .summary)
      LeftBound167786.bound (LeftBound167786.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69286⟩⟩) (rawTerms := some (Proof.Events655.exact167787RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound167786.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 167791)
      LeftBound167791.bound (LeftBound167791.actual selector witness) := by
  exact .transfer (LeftBound167791.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound167786.bound LeftBound167791.bound
def bound : CoeffClass := .finite ⟨32191361068277440720800338411520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound167786.bound, LeftBound167791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound167786.actual selector witness) * (LeftBound167791.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167792

namespace LeftBound167803
def owner : Owner := ⟨.program ⟨257⟩, ⟨68159⟩⟩
def transferEvent : Nat := 167803
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 167801 .coefficient) (.value (.predecessor 1 167802 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167801 .coefficient)
      LeftAuthority167799.bound (LeftAuthority167799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167802 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority167799.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167799.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority167799.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound167803

namespace LeftBound167807
def owner : Owner := ⟨.program ⟨257⟩, ⟨68160⟩⟩
def transferEvent : Nat := 167807
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 167805 .coefficient) (.predecessor 1 167806 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 167805 .coefficient)
      LeftBound163742.bound (LeftBound163742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 167806 .coefficient)
      LeftBound167803.bound (LeftBound167803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events655.exact167804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167803.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163742.bound LeftBound167803.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163742.bound, LeftBound167803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163742.actual selector witness) * (LeftBound167803.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167807

namespace LeftBound167808
def owner : Owner := ⟨.program ⟨257⟩, ⟨68160⟩⟩
def transferEvent : Nat := 167808
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68157⟩⟩]⟩ [⟨.result 167800 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167800 .coefficient)
      LeftAuthority167799.bound (LeftAuthority167799.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68157⟩⟩) (rawTerms := some (Proof.Events655.exact167800RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority167799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority167799.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority167799.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority167799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority167799.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound167808

namespace LeftBound167809
def owner : Owner := ⟨.program ⟨257⟩, ⟨68160⟩⟩
def transferEvent : Nat := 167809
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 163745 .summary) (.transfer 167808) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163745 .summary)
      LeftBound163743.bound (LeftBound163743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6466⟩⟩) (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 167808)
      LeftBound167808.bound (LeftBound167808.actual selector witness) := by
  exact .transfer (LeftBound167808.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163743.bound LeftBound167808.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163743.bound, LeftBound167808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163743.actual selector witness) * (LeftBound167808.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound167809

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
