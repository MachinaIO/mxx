import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard711

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound109206
def owner : Owner := ⟨.program ⟨257⟩, ⟨68931⟩⟩
def transferEvent : Nat := 109206
def frameStart : Nat := 109156
def rule : BoundRule := .sum [.predecessor 0 109204 .coefficient, .predecessor 1 109205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109204 .coefficient)
      LeftBound109189.bound (LeftBound109189.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound109189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109205 .coefficient)
      LeftAuthority109202.bound (LeftAuthority109202.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority109202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109189.bound, LeftAuthority109202.bound]
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109189.bound, LeftAuthority109202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109189.actual selector witness, LeftAuthority109202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109206

namespace LeftBound109209
def owner : Owner := ⟨.program ⟨257⟩, ⟨68932⟩⟩
def transferEvent : Nat := 109209
def frameStart : Nat := 109156
def rule : BoundRule := .identity (.predecessor 0 109208 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109208 .coefficient)
      LeftBound109206.bound (LeftBound109206.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound109206.derived selector witness)

def rawBound : CoeffClass := LeftBound109206.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound109206.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound109209

namespace LeftBound109215
def owner : Owner := ⟨.program ⟨257⟩, ⟨68933⟩⟩
def transferEvent : Nat := 109215
def frameStart : Nat := 109156
def rule : BoundRule := .product (.predecessor 0 109213 .coefficient) (.predecessor 1 109214 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109213 .coefficient)
      LeftAuthority109211.bound (LeftAuthority109211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109214 .coefficient)
      LeftBound109209.bound (LeftBound109209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109209.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority109211.bound LeftBound109209.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109211.bound, LeftBound109209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority109211.actual selector witness) * (LeftBound109209.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109215

namespace LeftBound109231
def owner : Owner := ⟨.program ⟨257⟩, ⟨9542⟩⟩
def transferEvent : Nat := 109231
def frameStart : Nat := 109156
def rule : BoundRule := .scale (.predecessor 0 109229 .coefficient) (.value (.predecessor 1 109230 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109229 .coefficient)
      LeftAuthority109227.bound (LeftAuthority109227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109227.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109230 .coefficient)
      LeftAuthority109218.bound (LeftAuthority109218.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority109218.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority109227.bound LeftAuthority109218.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109227.bound, LeftAuthority109218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority109227.actual selector witness) * (LeftAuthority109218.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound109231

namespace LeftBound109234
def owner : Owner := ⟨.program ⟨257⟩, ⟨7294⟩⟩
def transferEvent : Nat := 109234
def frameStart : Nat := 109156
def rule : BoundRule := .identity (.predecessor 0 109233 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109233 .coefficient)
      LeftAuthority109221.bound (LeftAuthority109221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109221.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109221.derived selector witness)

def rawBound : CoeffClass := LeftAuthority109221.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority109221.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound109234

namespace LeftBound109238
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def transferEvent : Nat := 109238
def frameStart : Nat := 109156
def rule : BoundRule := .product (.predecessor 0 109236 .coefficient) (.predecessor 1 109237 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109236 .coefficient)
      LeftBound109234.bound (LeftBound109234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109234.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109237 .coefficient)
      LeftBound109231.bound (LeftBound109231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound109234.bound LeftBound109231.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109234.bound, LeftBound109231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound109234.actual selector witness) * (LeftBound109231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109238

namespace LeftBound109243
def owner : Owner := ⟨.program ⟨257⟩, ⟨68934⟩⟩
def transferEvent : Nat := 109243
def frameStart : Nat := 109156
def rule : BoundRule := .sum [.predecessor 0 109241 .coefficient, .predecessor 1 109242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109241 .coefficient)
      LeftBound109238.bound (LeftBound109238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109242 .coefficient)
      LeftBound109215.bound (LeftBound109215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109215.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109238.bound, LeftBound109215.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109238.bound, LeftBound109215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109238.actual selector witness, LeftBound109215.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109243

namespace LeftBound109247
def owner : Owner := ⟨.program ⟨257⟩, ⟨69254⟩⟩
def transferEvent : Nat := 109247
def frameStart : Nat := 109156
def rule : BoundRule := .product (.predecessor 0 109245 .coefficient) (.predecessor 1 109246 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109245 .coefficient)
      LeftBound109243.bound (LeftBound109243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109246 .coefficient)
      LeftAuthority109200.bound (LeftAuthority109200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound109243.bound LeftAuthority109200.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109243.bound, LeftAuthority109200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound109243.actual selector witness) * (LeftAuthority109200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109247

namespace LeftBound109258
def owner : Owner := ⟨.program ⟨257⟩, ⟨65798⟩⟩
def transferEvent : Nat := 109258
def frameStart : Nat := 109156
def rule : BoundRule := .product (.predecessor 0 109256 .coefficient) (.predecessor 1 109257 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109256 .coefficient)
      LeftAuthority109211.bound (LeftAuthority109211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109257 .coefficient)
      LeftAuthority109254.bound (LeftAuthority109254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109254.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109254.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority109211.bound LeftAuthority109254.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109211.bound, LeftAuthority109254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority109211.actual selector witness) * (LeftAuthority109254.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109258

namespace LeftBound109266
def owner : Owner := ⟨.program ⟨257⟩, ⟨65799⟩⟩
def transferEvent : Nat := 109266
def frameStart : Nat := 109156
def rule : BoundRule := .sum [.predecessor 0 109264 .coefficient, .predecessor 1 109265 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109264 .coefficient)
      LeftAuthority109262.bound (LeftAuthority109262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109265 .coefficient)
      LeftBound109258.bound (LeftBound109258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109258.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority109262.bound, LeftBound109258.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109262.bound, LeftBound109258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority109262.actual selector witness, LeftBound109258.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109266

namespace LeftBound109270
def owner : Owner := ⟨.program ⟨257⟩, ⟨69255⟩⟩
def transferEvent : Nat := 109270
def frameStart : Nat := 109156
def rule : BoundRule := .sum [.predecessor 0 109268 .coefficient, .predecessor 1 109269 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109268 .coefficient)
      LeftBound109266.bound (LeftBound109266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109269 .coefficient)
      LeftBound109247.bound (LeftBound109247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109247.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109266.bound, LeftBound109247.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109266.bound, LeftBound109247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109266.actual selector witness, LeftBound109247.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109270

namespace LeftBound109283
def owner : Owner := ⟨.program ⟨257⟩, ⟨69253⟩⟩
def transferEvent : Nat := 109283
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 109281 .coefficient, .predecessor 1 109282 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109281 .coefficient)
      LeftBound109104.bound (LeftBound109104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109282 .coefficient)
      LeftBound109087.bound (LeftBound109087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109104.bound, LeftBound109087.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109104.bound, LeftBound109087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109104.actual selector witness, LeftBound109087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109283

namespace LeftBound109286
def owner : Owner := ⟨.program ⟨257⟩, ⟨69253⟩⟩
def transferEvent : Nat := 109286
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 109280 .summary, .result 109094 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109280 .summary)
      LeftBound109106.bound (LeftBound109106.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67783⟩⟩) (rawTerms := some (Proof.Events426.exact109280RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound109106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109094 .summary)
      LeftBound109089.bound (LeftBound109089.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69252⟩⟩) (rawTerms := some (Proof.Events426.exact109094RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound109089.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109106.bound, LeftBound109089.bound]
def bound : CoeffClass := .finite ⟨2998054127048462696448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109106.bound, LeftBound109089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109106.actual selector witness, LeftBound109089.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109286

namespace LeftBound109290
def owner : Owner := ⟨.program ⟨257⟩, ⟨70258⟩⟩
def transferEvent : Nat := 109290
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 109288 .coefficient) (.predecessor 1 109289 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109288 .coefficient)
      LeftBound109283.bound (LeftBound109283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events426.exact109287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109289 .coefficient)
      LeftAuthority109009.bound (LeftAuthority109009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events425.exact109010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109009.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109009.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound109283.bound LeftAuthority109009.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109283.bound, LeftAuthority109009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound109283.actual selector witness) * (LeftAuthority109009.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109290

namespace LeftBound109291
def owner : Owner := ⟨.program ⟨257⟩, ⟨70258⟩⟩
def transferEvent : Nat := 109291
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨70256⟩⟩]⟩ [⟨.result 109010 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109010 .coefficient)
      LeftAuthority109009.bound (LeftAuthority109009.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨70256⟩⟩) (rawTerms := some (Proof.Events425.exact109010RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109009.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109009.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority109009.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority109009.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound109291

namespace LeftBound109292
def owner : Owner := ⟨.program ⟨257⟩, ⟨70258⟩⟩
def transferEvent : Nat := 109292
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 109287 .summary) (.transfer 109291) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109287 .summary)
      LeftBound109286.bound (LeftBound109286.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69253⟩⟩) (rawTerms := some (Proof.Events426.exact109287RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound109286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 109291)
      LeftBound109291.bound (LeftBound109291.actual selector witness) := by
  exact .transfer (LeftBound109291.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound109286.bound LeftBound109291.bound
def bound : CoeffClass := .finite ⟨32191361068277440720800338411520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109286.bound, LeftBound109291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound109286.actual selector witness) * (LeftBound109291.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109292

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
