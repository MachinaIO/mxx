import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1328
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1372

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound205232
def owner : Owner := ⟨.program ⟨257⟩, ⟨62825⟩⟩
def transferEvent : Nat := 205232
def frameStart : Nat := 205193
def rule : BoundRule := .identity (.predecessor 0 205231 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205231 .coefficient)
      LeftAuthority205229.bound (LeftAuthority205229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority205229.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority205229.derived selector witness)

def rawBound : CoeffClass := LeftAuthority205229.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority205229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority205229.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound205232

namespace LeftBound205249
def owner : Owner := ⟨.program ⟨257⟩, ⟨64294⟩⟩
def transferEvent : Nat := 205249
def frameStart : Nat := 205193
def rule : BoundRule := .sum [.predecessor 0 205247 .coefficient, .predecessor 1 205248 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205247 .coefficient)
      LeftBound205232.bound (LeftBound205232.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound205232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205248 .coefficient)
      LeftAuthority205245.bound (LeftAuthority205245.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority205245.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound205232.bound, LeftAuthority205245.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound205232.bound, LeftAuthority205245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound205232.actual selector witness, LeftAuthority205245.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound205249

namespace LeftBound205252
def owner : Owner := ⟨.program ⟨257⟩, ⟨64295⟩⟩
def transferEvent : Nat := 205252
def frameStart : Nat := 205193
def rule : BoundRule := .identity (.predecessor 0 205251 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205251 .coefficient)
      LeftBound205249.bound (LeftBound205249.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound205249.derived selector witness)

def rawBound : CoeffClass := LeftBound205249.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound205249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound205249.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound205252

namespace LeftBound205258
def owner : Owner := ⟨.program ⟨257⟩, ⟨64296⟩⟩
def transferEvent : Nat := 205258
def frameStart : Nat := 205193
def rule : BoundRule := .product (.predecessor 0 205256 .coefficient) (.predecessor 1 205257 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205256 .coefficient)
      LeftAuthority205254.bound (LeftAuthority205254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority205254.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority205254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205257 .coefficient)
      LeftBound205252.bound (LeftBound205252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205252.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority205254.bound LeftBound205252.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority205254.bound, LeftBound205252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority205254.actual selector witness) * (LeftBound205252.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound205258

namespace LeftBound205266
def owner : Owner := ⟨.program ⟨257⟩, ⟨64297⟩⟩
def transferEvent : Nat := 205266
def frameStart : Nat := 205193
def rule : BoundRule := .sum [.predecessor 0 205264 .coefficient, .predecessor 1 205265 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205264 .coefficient)
      LeftAuthority205262.bound (LeftAuthority205262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority205262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority205262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205265 .coefficient)
      LeftBound205258.bound (LeftBound205258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205258.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority205262.bound, LeftBound205258.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority205262.bound, LeftBound205258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority205262.actual selector witness, LeftBound205258.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound205266

namespace LeftBound205270
def owner : Owner := ⟨.program ⟨257⟩, ⟨64928⟩⟩
def transferEvent : Nat := 205270
def frameStart : Nat := 205193
def rule : BoundRule := .product (.predecessor 0 205268 .coefficient) (.predecessor 1 205269 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205268 .coefficient)
      LeftBound205266.bound (LeftBound205266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205269 .coefficient)
      LeftAuthority205243.bound (LeftAuthority205243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority205243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority205243.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound205266.bound LeftAuthority205243.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound205266.bound, LeftAuthority205243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound205266.actual selector witness) * (LeftAuthority205243.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound205270

namespace LeftBound205281
def owner : Owner := ⟨.program ⟨257⟩, ⟨63126⟩⟩
def transferEvent : Nat := 205281
def frameStart : Nat := 205193
def rule : BoundRule := .product (.predecessor 0 205279 .coefficient) (.predecessor 1 205280 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205279 .coefficient)
      LeftAuthority205254.bound (LeftAuthority205254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority205254.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority205254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205280 .coefficient)
      LeftAuthority205277.bound (LeftAuthority205277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority205277.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority205277.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority205254.bound LeftAuthority205277.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority205254.bound, LeftAuthority205277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority205254.actual selector witness) * (LeftAuthority205277.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound205281

namespace LeftBound205289
def owner : Owner := ⟨.program ⟨257⟩, ⟨63127⟩⟩
def transferEvent : Nat := 205289
def frameStart : Nat := 205193
def rule : BoundRule := .sum [.predecessor 0 205287 .coefficient, .predecessor 1 205288 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205287 .coefficient)
      LeftAuthority205285.bound (LeftAuthority205285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority205285.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority205285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205288 .coefficient)
      LeftBound205281.bound (LeftBound205281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority205285.bound, LeftBound205281.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority205285.bound, LeftBound205281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority205285.actual selector witness, LeftBound205281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound205289

namespace LeftBound205293
def owner : Owner := ⟨.program ⟨257⟩, ⟨64933⟩⟩
def transferEvent : Nat := 205293
def frameStart : Nat := 205193
def rule : BoundRule := .sum [.predecessor 0 205291 .coefficient, .predecessor 1 205292 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205291 .coefficient)
      LeftBound205289.bound (LeftBound205289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205292 .coefficient)
      LeftBound205270.bound (LeftBound205270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205270.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound205289.bound, LeftBound205270.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound205289.bound, LeftBound205270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound205289.actual selector witness, LeftBound205270.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound205293

namespace LeftBound205306
def owner : Owner := ⟨.program ⟨257⟩, ⟨64930⟩⟩
def transferEvent : Nat := 205306
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 205304 .coefficient, .predecessor 1 205305 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205304 .coefficient)
      LeftBound205135.bound (LeftBound205135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205305 .coefficient)
      LeftBound205118.bound (LeftBound205118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205118.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound205135.bound, LeftBound205118.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound205135.bound, LeftBound205118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound205135.actual selector witness, LeftBound205118.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound205306

namespace LeftBound205309
def owner : Owner := ⟨.program ⟨257⟩, ⟨64930⟩⟩
def transferEvent : Nat := 205309
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 205303 .summary, .result 205125 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 205303 .summary)
      LeftBound205137.bound (LeftBound205137.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63715⟩⟩) (rawTerms := some (Proof.Events801.exact205303RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound205137.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 205125 .summary)
      LeftBound205120.bound (LeftBound205120.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64929⟩⟩) (rawTerms := some (Proof.Events801.exact205125RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound205120.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound205137.bound, LeftBound205120.bound]
def bound : CoeffClass := .finite ⟨32190771716940580661919523012608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound205137.bound, LeftBound205120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound205137.actual selector witness, LeftBound205120.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound205309

namespace LeftBound205313
def owner : Owner := ⟨.program ⟨257⟩, ⟨64931⟩⟩
def transferEvent : Nat := 205313
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 205311 .coefficient) (.predecessor 1 205312 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205311 .coefficient)
      LeftBound205306.bound (LeftBound205306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205312 .coefficient)
      LeftBound15721.bound (LeftBound15721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15721.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound205306.bound LeftBound15721.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound205306.bound, LeftBound15721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound205306.actual selector witness) * (LeftBound15721.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound205313

namespace LeftBound205314
def owner : Owner := ⟨.program ⟨257⟩, ⟨64931⟩⟩
def transferEvent : Nat := 205314
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩ [⟨.result 15718 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15718 .coefficient)
      LeftAuthority15717.bound (LeftAuthority15717.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7099⟩⟩) (rawTerms := some (Proof.Events061.exact15718RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15717.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15717.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15717.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15717.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound205314

namespace LeftBound205315
def owner : Owner := ⟨.program ⟨257⟩, ⟨64931⟩⟩
def transferEvent : Nat := 205315
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 205310 .summary) (.transfer 205314) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 205310 .summary)
      LeftBound205309.bound (LeftBound205309.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64930⟩⟩) (rawTerms := some (Proof.Events801.exact205310RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound205309.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 205314)
      LeftBound205314.bound (LeftBound205314.actual selector witness) := by
  exact .transfer (LeftBound205314.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound205309.bound LeftBound205314.bound
def bound : CoeffClass := .finite ⟨345645779393153907795485959807676889169920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound205309.bound, LeftBound205314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound205309.actual selector witness) * (LeftBound205314.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound205315

namespace LeftBound205330
def owner : Owner := ⟨.program ⟨257⟩, ⟨61949⟩⟩
def transferEvent : Nat := 205330
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 205328 .coefficient) (.predecessor 1 205329 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 205328 .coefficient)
      LeftBound197997.bound (LeftBound197997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events773.exact198001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197997.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 205329 .coefficient)
      LeftAuthority205326.bound (LeftAuthority205326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events802.exact205327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority205326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority205326.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound197997.bound LeftAuthority205326.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound197997.bound, LeftAuthority205326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound197997.actual selector witness) * (LeftAuthority205326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound205330

namespace LeftBound205331
def owner : Owner := ⟨.program ⟨257⟩, ⟨61949⟩⟩
def transferEvent : Nat := 205331
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨61947⟩⟩]⟩ [⟨.result 205327 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 205327 .coefficient)
      LeftAuthority205326.bound (LeftAuthority205326.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨61947⟩⟩) (rawTerms := some (Proof.Events802.exact205327RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority205326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority205326.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority205326.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority205326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority205326.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound205331

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
