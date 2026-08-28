import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard106
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1185
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1188
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1223

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound182916
def owner : Owner := ⟨.program ⟨257⟩, ⟨63739⟩⟩
def transferEvent : Nat := 182916
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 178370 .summary) (.transfer 182915) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178370 .summary)
      LeftBound178368.bound (LeftBound178368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6186⟩⟩) (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 182915)
      LeftBound182915.bound (LeftBound182915.actual selector witness) := by
  exact .transfer (LeftBound182915.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178368.bound LeftBound182915.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178368.bound, LeftBound182915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178368.actual selector witness) * (LeftBound182915.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound182916

namespace LeftBound183011
def owner : Owner := ⟨.program ⟨257⟩, ⟨62833⟩⟩
def transferEvent : Nat := 183011
def frameStart : Nat := 182972
def rule : BoundRule := .identity (.predecessor 0 183010 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183010 .coefficient)
      LeftAuthority183008.bound (LeftAuthority183008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events714.exact183009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183008.derived selector witness)

def rawBound : CoeffClass := LeftAuthority183008.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority183008.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound183011

namespace LeftBound183028
def owner : Owner := ⟨.program ⟨257⟩, ⟨64298⟩⟩
def transferEvent : Nat := 183028
def frameStart : Nat := 182972
def rule : BoundRule := .sum [.predecessor 0 183026 .coefficient, .predecessor 1 183027 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183026 .coefficient)
      LeftBound183011.bound (LeftBound183011.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound183011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183027 .coefficient)
      LeftAuthority183024.bound (LeftAuthority183024.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority183024.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183011.bound, LeftAuthority183024.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183011.bound, LeftAuthority183024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183011.actual selector witness, LeftAuthority183024.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183028

namespace LeftBound183031
def owner : Owner := ⟨.program ⟨257⟩, ⟨64299⟩⟩
def transferEvent : Nat := 183031
def frameStart : Nat := 182972
def rule : BoundRule := .identity (.predecessor 0 183030 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183030 .coefficient)
      LeftBound183028.bound (LeftBound183028.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound183028.derived selector witness)

def rawBound : CoeffClass := LeftBound183028.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound183028.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound183031

namespace LeftBound183037
def owner : Owner := ⟨.program ⟨257⟩, ⟨64300⟩⟩
def transferEvent : Nat := 183037
def frameStart : Nat := 182972
def rule : BoundRule := .product (.predecessor 0 183035 .coefficient) (.predecessor 1 183036 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183035 .coefficient)
      LeftAuthority183033.bound (LeftAuthority183033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events714.exact183034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183033.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183036 .coefficient)
      LeftBound183031.bound (LeftBound183031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events714.exact183032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183031.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority183033.bound LeftBound183031.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183033.bound, LeftBound183031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority183033.actual selector witness) * (LeftBound183031.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183037

namespace LeftBound183045
def owner : Owner := ⟨.program ⟨257⟩, ⟨64301⟩⟩
def transferEvent : Nat := 183045
def frameStart : Nat := 182972
def rule : BoundRule := .sum [.predecessor 0 183043 .coefficient, .predecessor 1 183044 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183043 .coefficient)
      LeftAuthority183041.bound (LeftAuthority183041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183041.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183044 .coefficient)
      LeftBound183037.bound (LeftBound183037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events714.exact183039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183037.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183037.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority183041.bound, LeftBound183037.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183041.bound, LeftBound183037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority183041.actual selector witness, LeftBound183037.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183045

namespace LeftBound183049
def owner : Owner := ⟨.program ⟨257⟩, ⟨64966⟩⟩
def transferEvent : Nat := 183049
def frameStart : Nat := 182972
def rule : BoundRule := .product (.predecessor 0 183047 .coefficient) (.predecessor 1 183048 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183047 .coefficient)
      LeftBound183045.bound (LeftBound183045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183045.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183048 .coefficient)
      LeftAuthority183022.bound (LeftAuthority183022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events714.exact183023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183022.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound183045.bound LeftAuthority183022.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183045.bound, LeftAuthority183022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound183045.actual selector witness) * (LeftAuthority183022.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183049

namespace LeftBound183060
def owner : Owner := ⟨.program ⟨257⟩, ⟨63140⟩⟩
def transferEvent : Nat := 183060
def frameStart : Nat := 182972
def rule : BoundRule := .product (.predecessor 0 183058 .coefficient) (.predecessor 1 183059 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183058 .coefficient)
      LeftAuthority183033.bound (LeftAuthority183033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events714.exact183034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183033.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183059 .coefficient)
      LeftAuthority183056.bound (LeftAuthority183056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183056.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority183033.bound LeftAuthority183056.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183033.bound, LeftAuthority183056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority183033.actual selector witness) * (LeftAuthority183056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183060

namespace LeftBound183068
def owner : Owner := ⟨.program ⟨257⟩, ⟨63141⟩⟩
def transferEvent : Nat := 183068
def frameStart : Nat := 182972
def rule : BoundRule := .sum [.predecessor 0 183066 .coefficient, .predecessor 1 183067 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183066 .coefficient)
      LeftAuthority183064.bound (LeftAuthority183064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183064.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183064.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183067 .coefficient)
      LeftBound183060.bound (LeftBound183060.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183062RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183060.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183060.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority183064.bound, LeftBound183060.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183064.bound, LeftBound183060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority183064.actual selector witness, LeftBound183060.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183068

namespace LeftBound183072
def owner : Owner := ⟨.program ⟨257⟩, ⟨64970⟩⟩
def transferEvent : Nat := 183072
def frameStart : Nat := 182972
def rule : BoundRule := .sum [.predecessor 0 183070 .coefficient, .predecessor 1 183071 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183070 .coefficient)
      LeftBound183068.bound (LeftBound183068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183071 .coefficient)
      LeftBound183049.bound (LeftBound183049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183049.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183068.bound, LeftBound183049.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183068.bound, LeftBound183049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183068.actual selector witness, LeftBound183049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183072

namespace LeftBound183085
def owner : Owner := ⟨.program ⟨257⟩, ⟨64968⟩⟩
def transferEvent : Nat := 183085
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 183083 .coefficient, .predecessor 1 183084 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183083 .coefficient)
      LeftBound182914.bound (LeftBound182914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183084 .coefficient)
      LeftBound182897.bound (LeftBound182897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events714.exact182904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182897.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound182914.bound, LeftBound182897.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound182914.bound, LeftBound182897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound182914.actual selector witness, LeftBound182897.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183085

namespace LeftBound183088
def owner : Owner := ⟨.program ⟨257⟩, ⟨64968⟩⟩
def transferEvent : Nat := 183088
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 183082 .summary, .result 182904 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183082 .summary)
      LeftBound182916.bound (LeftBound182916.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63739⟩⟩) (rawTerms := some (Proof.Events715.exact183082RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound182916.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 182904 .summary)
      LeftBound182899.bound (LeftBound182899.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64967⟩⟩) (rawTerms := some (Proof.Events714.exact182904RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound182899.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound182916.bound, LeftBound182899.bound]
def bound : CoeffClass := .finite ⟨32190771716940580661919523012608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound182916.bound, LeftBound182899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound182916.actual selector witness, LeftBound182899.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183088

namespace LeftBound183112
def owner : Owner := ⟨.program ⟨257⟩, ⟨25287⟩⟩
def transferEvent : Nat := 183112
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 183110 .coefficient) (.predecessor 1 183111 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183110 .coefficient)
      LeftAuthority8551.bound (LeftAuthority8551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183111 .coefficient)
      LeftBound178276.bound (LeftBound178276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority8551.bound LeftBound178276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8551.bound, LeftBound178276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority8551.actual selector witness) * (LeftBound178276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound183112

namespace LeftBound183117
def owner : Owner := ⟨.program ⟨257⟩, ⟨8922⟩⟩
def transferEvent : Nat := 183117
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 183115 .coefficient) (.predecessor 1 183116 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183115 .coefficient)
      LeftBound178147.bound (LeftBound178147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183116 .coefficient)
      LeftBound22089.bound (LeftBound22089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22089.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22089.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound178147.bound LeftBound22089.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178147.bound, LeftBound22089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound178147.actual selector witness) * (LeftBound22089.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183117

namespace LeftBound183122
def owner : Owner := ⟨.program ⟨257⟩, ⟨25288⟩⟩
def transferEvent : Nat := 183122
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 183120 .coefficient, .predecessor 1 183121 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183120 .coefficient)
      LeftBound183117.bound (LeftBound183117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183121 .coefficient)
      LeftBound183112.bound (LeftBound183112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183112.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183117.bound, LeftBound183112.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183117.bound, LeftBound183112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183117.actual selector witness, LeftBound183112.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183122

namespace LeftBound183126
def owner : Owner := ⟨.program ⟨257⟩, ⟨25289⟩⟩
def transferEvent : Nat := 183126
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 183124 .coefficient, .predecessor 1 183125 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183124 .coefficient)
      LeftBound183122.bound (LeftBound183122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183125 .coefficient)
      LeftBound22081.bound (LeftBound22081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22081.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183122.bound, LeftBound22081.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183122.bound, LeftBound22081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183122.actual selector witness, LeftBound22081.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183126

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
