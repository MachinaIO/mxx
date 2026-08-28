import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard209
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard255

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound44165
def owner : Owner := ⟨.program ⟨257⟩, ⟨69044⟩⟩
def transferEvent : Nat := 44165
def frameStart : Nat := 44106
def rule : BoundRule := .identity (.predecessor 0 44164 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44164 .coefficient)
      LeftBound44162.bound (LeftBound44162.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound44162.derived selector witness)

def rawBound : CoeffClass := LeftBound44162.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound44162.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound44165

namespace LeftBound44171
def owner : Owner := ⟨.program ⟨257⟩, ⟨69045⟩⟩
def transferEvent : Nat := 44171
def frameStart : Nat := 44106
def rule : BoundRule := .product (.predecessor 0 44169 .coefficient) (.predecessor 1 44170 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44169 .coefficient)
      LeftAuthority44167.bound (LeftAuthority44167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44170 .coefficient)
      LeftBound44165.bound (LeftBound44165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44165.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority44167.bound LeftBound44165.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44167.bound, LeftBound44165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority44167.actual selector witness) * (LeftBound44165.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44171

namespace LeftBound44179
def owner : Owner := ⟨.program ⟨257⟩, ⟨69046⟩⟩
def transferEvent : Nat := 44179
def frameStart : Nat := 44106
def rule : BoundRule := .sum [.predecessor 0 44177 .coefficient, .predecessor 1 44178 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44177 .coefficient)
      LeftAuthority44175.bound (LeftAuthority44175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44178 .coefficient)
      LeftBound44171.bound (LeftBound44171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44171.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority44175.bound, LeftBound44171.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44175.bound, LeftBound44171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority44175.actual selector witness, LeftBound44171.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44179

namespace LeftBound44183
def owner : Owner := ⟨.program ⟨257⟩, ⟨70874⟩⟩
def transferEvent : Nat := 44183
def frameStart : Nat := 44106
def rule : BoundRule := .product (.predecessor 0 44181 .coefficient) (.predecessor 1 44182 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44181 .coefficient)
      LeftBound44179.bound (LeftBound44179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44182 .coefficient)
      LeftAuthority44156.bound (LeftAuthority44156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44156.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44156.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound44179.bound LeftAuthority44156.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44179.bound, LeftAuthority44156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound44179.actual selector witness) * (LeftAuthority44156.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44183

namespace LeftBound44194
def owner : Owner := ⟨.program ⟨257⟩, ⟨67229⟩⟩
def transferEvent : Nat := 44194
def frameStart : Nat := 44106
def rule : BoundRule := .product (.predecessor 0 44192 .coefficient) (.predecessor 1 44193 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44192 .coefficient)
      LeftAuthority44167.bound (LeftAuthority44167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44193 .coefficient)
      LeftAuthority44190.bound (LeftAuthority44190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44190.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority44167.bound LeftAuthority44190.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44167.bound, LeftAuthority44190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority44167.actual selector witness) * (LeftAuthority44190.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44194

namespace LeftBound44202
def owner : Owner := ⟨.program ⟨257⟩, ⟨67230⟩⟩
def transferEvent : Nat := 44202
def frameStart : Nat := 44106
def rule : BoundRule := .sum [.predecessor 0 44200 .coefficient, .predecessor 1 44201 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44200 .coefficient)
      LeftAuthority44198.bound (LeftAuthority44198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44201 .coefficient)
      LeftBound44194.bound (LeftBound44194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority44198.bound, LeftBound44194.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44198.bound, LeftBound44194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority44198.actual selector witness, LeftBound44194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44202

namespace LeftBound44206
def owner : Owner := ⟨.program ⟨257⟩, ⟨70887⟩⟩
def transferEvent : Nat := 44206
def frameStart : Nat := 44106
def rule : BoundRule := .sum [.predecessor 0 44204 .coefficient, .predecessor 1 44205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44204 .coefficient)
      LeftBound44202.bound (LeftBound44202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44205 .coefficient)
      LeftBound44183.bound (LeftBound44183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44183.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44183.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44202.bound, LeftBound44183.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44202.bound, LeftBound44183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound44202.actual selector witness, LeftBound44183.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44206

namespace LeftBound44219
def owner : Owner := ⟨.program ⟨257⟩, ⟨70876⟩⟩
def transferEvent : Nat := 44219
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44217 .coefficient, .predecessor 1 44218 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44217 .coefficient)
      LeftBound44048.bound (LeftBound44048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44218 .coefficient)
      LeftBound44031.bound (LeftBound44031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44048.bound, LeftBound44031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44048.bound, LeftBound44031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound44048.actual selector witness, LeftBound44031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44219

namespace LeftBound44222
def owner : Owner := ⟨.program ⟨257⟩, ⟨70876⟩⟩
def transferEvent : Nat := 44222
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44216 .summary, .result 44038 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44216 .summary)
      LeftBound44050.bound (LeftBound44050.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68256⟩⟩) (rawTerms := some (Proof.Events172.exact44216RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44038 .summary)
      LeftBound44033.bound (LeftBound44033.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70875⟩⟩) (rawTerms := some (Proof.Events172.exact44038RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44033.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44050.bound, LeftBound44033.bound]
def bound : CoeffClass := .finite ⟨32191361068277642793642192273408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44050.bound, LeftBound44033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound44050.actual selector witness, LeftBound44033.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44222

namespace LeftBound44226
def owner : Owner := ⟨.program ⟨257⟩, ⟨70877⟩⟩
def transferEvent : Nat := 44226
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44224 .coefficient) (.predecessor 1 44225 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44224 .coefficient)
      LeftBound44219.bound (LeftBound44219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44225 .coefficient)
      LeftBound15701.bound (LeftBound15701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15701.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound44219.bound LeftBound15701.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44219.bound, LeftBound15701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound44219.actual selector witness) * (LeftBound15701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44226

namespace LeftBound44227
def owner : Owner := ⟨.program ⟨257⟩, ⟨70877⟩⟩
def transferEvent : Nat := 44227
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩ [⟨.result 15698 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15698 .coefficient)
      LeftAuthority15697.bound (LeftAuthority15697.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7173⟩⟩) (rawTerms := some (Proof.Events061.exact15698RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15697.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15697.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15697.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15697.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44227

namespace LeftBound44228
def owner : Owner := ⟨.program ⟨257⟩, ⟨70877⟩⟩
def transferEvent : Nat := 44228
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 44223 .summary) (.transfer 44227) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44223 .summary)
      LeftBound44222.bound (LeftBound44222.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70876⟩⟩) (rawTerms := some (Proof.Events172.exact44223RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 44227)
      LeftBound44227.bound (LeftBound44227.actual selector witness) := by
  exact .transfer (LeftBound44227.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound44222.bound LeftBound44227.bound
def bound : CoeffClass := .finite ⟨345652107504950247116658231350078126161920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44222.bound, LeftBound44227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound44222.actual selector witness) * (LeftBound44227.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44228

namespace LeftBound44243
def owner : Owner := ⟨.program ⟨257⟩, ⟨65146⟩⟩
def transferEvent : Nat := 44243
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44241 .coefficient) (.predecessor 1 44242 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44241 .coefficient)
      LeftBound36640.bound (LeftBound36640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36640.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44242 .coefficient)
      LeftAuthority44239.bound (LeftAuthority44239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44239.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound36640.bound LeftAuthority44239.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36640.bound, LeftAuthority44239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound36640.actual selector witness) * (LeftAuthority44239.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44243

namespace LeftBound44244
def owner : Owner := ⟨.program ⟨257⟩, ⟨65146⟩⟩
def transferEvent : Nat := 44244
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨65144⟩⟩]⟩ [⟨.result 44240 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44240 .coefficient)
      LeftAuthority44239.bound (LeftAuthority44239.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨65144⟩⟩) (rawTerms := some (Proof.Events172.exact44240RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44239.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority44239.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority44239.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44244

namespace LeftBound44245
def owner : Owner := ⟨.program ⟨257⟩, ⟨65146⟩⟩
def transferEvent : Nat := 44245
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36644 .summary) (.transfer 44244) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36644 .summary)
      LeftBound36643.bound (LeftBound36643.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64540⟩⟩) (rawTerms := some (Proof.Events143.exact36644RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 44244)
      LeftBound44244.bound (LeftBound44244.actual selector witness) := by
  exact .transfer (LeftBound44244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound36643.bound LeftBound44244.bound
def bound : CoeffClass := .finite ⟨32190771716940378589077669150720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36643.bound, LeftBound44244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound36643.actual selector witness) * (LeftBound44244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44245

namespace LeftBound44256
def owner : Owner := ⟨.program ⟨257⟩, ⟨63854⟩⟩
def transferEvent : Nat := 44256
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 44254 .coefficient) (.value (.predecessor 1 44255 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44254 .coefficient)
      LeftAuthority44252.bound (LeftAuthority44252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44252.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 44255 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority44252.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44252.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority44252.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound44256

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
