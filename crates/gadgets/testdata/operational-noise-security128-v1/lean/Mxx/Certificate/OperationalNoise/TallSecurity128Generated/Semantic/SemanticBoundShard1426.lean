import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1392
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1425

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound212095
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def transferEvent : Nat := 212095
def frameStart : Nat := 212013
def rule : BoundRule := .product (.predecessor 0 212093 .coefficient) (.predecessor 1 212094 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212093 .coefficient)
      LeftBound212091.bound (LeftBound212091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212094 .coefficient)
      LeftBound212088.bound (LeftBound212088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212088.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212088.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound212091.bound LeftBound212088.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212091.bound, LeftBound212088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound212091.actual selector witness) * (LeftBound212088.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212095

namespace LeftBound212100
def owner : Owner := ⟨.program ⟨257⟩, ⟨64209⟩⟩
def transferEvent : Nat := 212100
def frameStart : Nat := 212013
def rule : BoundRule := .sum [.predecessor 0 212098 .coefficient, .predecessor 1 212099 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212098 .coefficient)
      LeftBound212095.bound (LeftBound212095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212099 .coefficient)
      LeftBound212072.bound (LeftBound212072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212072.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound212095.bound, LeftBound212072.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212095.bound, LeftBound212072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound212095.actual selector witness, LeftBound212072.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212100

namespace LeftBound212104
def owner : Owner := ⟨.program ⟨257⟩, ⟨64442⟩⟩
def transferEvent : Nat := 212104
def frameStart : Nat := 212013
def rule : BoundRule := .product (.predecessor 0 212102 .coefficient) (.predecessor 1 212103 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212102 .coefficient)
      LeftBound212100.bound (LeftBound212100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212103 .coefficient)
      LeftAuthority212057.bound (LeftAuthority212057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212057.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound212100.bound LeftAuthority212057.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212100.bound, LeftAuthority212057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound212100.actual selector witness) * (LeftAuthority212057.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212104

namespace LeftBound212115
def owner : Owner := ⟨.program ⟨257⟩, ⟨62810⟩⟩
def transferEvent : Nat := 212115
def frameStart : Nat := 212013
def rule : BoundRule := .product (.predecessor 0 212113 .coefficient) (.predecessor 1 212114 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212113 .coefficient)
      LeftAuthority212068.bound (LeftAuthority212068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212068.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212114 .coefficient)
      LeftAuthority212111.bound (LeftAuthority212111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212111.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212111.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority212068.bound LeftAuthority212111.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority212068.bound, LeftAuthority212111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority212068.actual selector witness) * (LeftAuthority212111.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212115

namespace LeftBound212123
def owner : Owner := ⟨.program ⟨257⟩, ⟨62811⟩⟩
def transferEvent : Nat := 212123
def frameStart : Nat := 212013
def rule : BoundRule := .sum [.predecessor 0 212121 .coefficient, .predecessor 1 212122 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212121 .coefficient)
      LeftAuthority212119.bound (LeftAuthority212119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212122 .coefficient)
      LeftBound212115.bound (LeftBound212115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212115.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212115.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority212119.bound, LeftBound212115.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority212119.bound, LeftBound212115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority212119.actual selector witness, LeftBound212115.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212123

namespace LeftBound212127
def owner : Owner := ⟨.program ⟨257⟩, ⟨64443⟩⟩
def transferEvent : Nat := 212127
def frameStart : Nat := 212013
def rule : BoundRule := .sum [.predecessor 0 212125 .coefficient, .predecessor 1 212126 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212125 .coefficient)
      LeftBound212123.bound (LeftBound212123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212123.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212126 .coefficient)
      LeftBound212104.bound (LeftBound212104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212104.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound212123.bound, LeftBound212104.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212123.bound, LeftBound212104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound212123.actual selector witness, LeftBound212104.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212127

namespace LeftBound212140
def owner : Owner := ⟨.program ⟨257⟩, ⟨64441⟩⟩
def transferEvent : Nat := 212140
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 212138 .coefficient, .predecessor 1 212139 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212138 .coefficient)
      LeftBound211961.bound (LeftBound211961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212139 .coefficient)
      LeftBound211944.bound (LeftBound211944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events827.exact211951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211944.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound211961.bound, LeftBound211944.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound211961.bound, LeftBound211944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound211961.actual selector witness, LeftBound211944.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212140

namespace LeftBound212143
def owner : Owner := ⟨.program ⟨257⟩, ⟨64441⟩⟩
def transferEvent : Nat := 212143
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 212137 .summary, .result 211951 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 212137 .summary)
      LeftBound211963.bound (LeftBound211963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63372⟩⟩) (rawTerms := some (Proof.Events828.exact212137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound211963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 211951 .summary)
      LeftBound211946.bound (LeftBound211946.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64440⟩⟩) (rawTerms := some (Proof.Events827.exact211951RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound211946.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound211963.bound, LeftBound211946.bound]
def bound : CoeffClass := .finite ⟨2997999239428004118528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound211963.bound, LeftBound211946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound211963.actual selector witness, LeftBound211946.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound212143

namespace LeftBound212147
def owner : Owner := ⟨.program ⟨257⟩, ⟨64874⟩⟩
def transferEvent : Nat := 212147
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 212145 .coefficient) (.predecessor 1 212146 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212145 .coefficient)
      LeftBound212140.bound (LeftBound212140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212146 .coefficient)
      LeftAuthority211866.bound (LeftAuthority211866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events827.exact211867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority211866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority211866.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound212140.bound LeftAuthority211866.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212140.bound, LeftAuthority211866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound212140.actual selector witness) * (LeftAuthority211866.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212147

namespace LeftBound212148
def owner : Owner := ⟨.program ⟨257⟩, ⟨64874⟩⟩
def transferEvent : Nat := 212148
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨64872⟩⟩]⟩ [⟨.result 211867 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 211867 .coefficient)
      LeftAuthority211866.bound (LeftAuthority211866.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨64872⟩⟩) (rawTerms := some (Proof.Events827.exact211867RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority211866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority211866.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority211866.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority211866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority211866.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound212148

namespace LeftBound212149
def owner : Owner := ⟨.program ⟨257⟩, ⟨64874⟩⟩
def transferEvent : Nat := 212149
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 212144 .summary) (.transfer 212148) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 212144 .summary)
      LeftBound212143.bound (LeftBound212143.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64441⟩⟩) (rawTerms := some (Proof.Events828.exact212144RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound212143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 212148)
      LeftBound212148.bound (LeftBound212148.actual selector witness) := by
  exact .transfer (LeftBound212148.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound212143.bound LeftBound212148.bound
def bound : CoeffClass := .finite ⟨32190771716940378589077669150720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound212143.bound, LeftBound212148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound212143.actual selector witness) * (LeftBound212148.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212149

namespace LeftBound212160
def owner : Owner := ⟨.program ⟨257⟩, ⟨63678⟩⟩
def transferEvent : Nat := 212160
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 212158 .coefficient) (.value (.predecessor 1 212159 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212158 .coefficient)
      LeftAuthority212156.bound (LeftAuthority212156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212156.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212159 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority212156.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority212156.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority212156.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound212160

namespace LeftBound212164
def owner : Owner := ⟨.program ⟨257⟩, ⟨63679⟩⟩
def transferEvent : Nat := 212164
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 212162 .coefficient) (.predecessor 1 212163 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212162 .coefficient)
      LeftBound207617.bound (LeftBound207617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 212163 .coefficient)
      LeftBound212160.bound (LeftBound212160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events828.exact212161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212160.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207617.bound LeftBound212160.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207617.bound, LeftBound212160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207617.actual selector witness) * (LeftBound212160.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212164

namespace LeftBound212165
def owner : Owner := ⟨.program ⟨257⟩, ⟨63679⟩⟩
def transferEvent : Nat := 212165
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨63676⟩⟩]⟩ [⟨.result 212157 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 212157 .coefficient)
      LeftAuthority212156.bound (LeftAuthority212156.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨63676⟩⟩) (rawTerms := some (Proof.Events828.exact212157RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212156.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212156.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority212156.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority212156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority212156.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound212165

namespace LeftBound212166
def owner : Owner := ⟨.program ⟨257⟩, ⟨63679⟩⟩
def transferEvent : Nat := 212166
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 207620 .summary) (.transfer 212165) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207620 .summary)
      LeftBound207618.bound (LeftBound207618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5599⟩⟩) (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 212165)
      LeftBound212165.bound (LeftBound212165.actual selector witness) := by
  exact .transfer (LeftBound212165.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207618.bound LeftBound212165.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207618.bound, LeftBound212165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207618.actual selector witness) * (LeftBound212165.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound212166

namespace LeftBound212261
def owner : Owner := ⟨.program ⟨257⟩, ⟨62809⟩⟩
def transferEvent : Nat := 212261
def frameStart : Nat := 212222
def rule : BoundRule := .identity (.predecessor 0 212260 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 212260 .coefficient)
      LeftAuthority212258.bound (LeftAuthority212258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority212258.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority212258.derived selector witness)

def rawBound : CoeffClass := LeftAuthority212258.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority212258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority212258.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound212261

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
